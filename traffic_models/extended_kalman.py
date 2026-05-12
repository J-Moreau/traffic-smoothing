from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
import scipy.linalg
from filterpy.common import outer_product_sum, pretty_str
from filterpy.kalman import EnsembleKalmanFilter
from numpy.random import multivariate_normal
from numpy.typing import NDArray

from traffic_models.flows import GreenshieldsFlow, QuadraticLinearFlow, TriangularFlow
from traffic_models.godunov import godunov_jacobian, godunov_step
from traffic_models.kalman import get_measure_at_timestep
from traffic_models.sim import DiscretizationGrid


@dataclass
class KalmanResult:
    rho_hat: np.ndarray
    v_hat: np.ndarray
    v_pred: np.ndarray
    std_rho: np.ndarray
    std_speed: np.ndarray
    cov_pred: np.ndarray
    cov: np.ndarray
    jacobian: np.ndarray
    params: Optional[np.ndarray] = None


def extended_kalman_step(
    x_pred: NDArray,
    y: NDArray,
    P: NDArray,
    F: NDArray,
    y_pred: NDArray,
    H: NDArray,
    Q: NDArray,
    R: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Extended Kalman filter assuming non linear state transition x^{t+1} = f(x^t)
    with jacobian F and non linear measurements y = h(x) with jacobian H

    using Cholesky decomposition for stability.

    Parameters:
        x_pred:   next state estimate x^{t+1|t} (n x 1)
        y:   measurement y^{t+1} (m x 1)
        P:   covariance estimate (n x n)
        F:   state transition jacobian (n x n)
        y_pred: predicted measurement y^{t+1|t} = h(x^{t+1|t})
        H:   measurement matrix (m x n)
        Q:   process noise covariance (n x n)
        R:   measurement noise covariance (m x m)

    Returns:
        x_corrected: updated state estimate
        y_pred: measurement estimate
        P_pred: covariance estimate
    """

    # === Predict ===
    P = F @ P @ F.T + Q

    # === Update ===
    S = H @ P @ H.T + R  # Innovation covariance

    # # Method 1: direct inverse
    # K = P @ H.T @ np.linalg.inv(S)

    # # Method 2: solve n systems
    # # K = P @ H.T @ S^-1
    # # K.T = S^-1 @ H @ P.T
    # # S K.T = H @ P
    # K = np.linalg.solve(S, H @ P).T

    # # Method 3: Cholesky
    L = np.linalg.cholesky(S)  # Cholesky: S = L @ L.T
    # # Solve for K using forward and backward substitution
    # # L L^T K.T = H P
    u = scipy.linalg.solve_triangular(L, H @ P, lower=True)  # L (L.T K.T) = H @ P
    K = scipy.linalg.solve_triangular(L.T, u).T  # L.T K.T = u


    x_corrected = x_pred + K @ (y - y_pred)  # State update
    # P_pred = (np.eye(x_pred.shape[0]) - K @ H) @ P  # Covariance update
    # Joseph form for numerical stability (more expensive)
    I = np.eye(P.shape[0])
    P_pred = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T 
    return x_corrected, y_pred, P_pred


def run_ekf_on_trajectories(
    discretized_trajectories: pl.DataFrame,
    P_0 : NDArray,
    v_0: NDArray | float,
    flow: GreenshieldsFlow|TriangularFlow|QuadraticLinearFlow,
    Q: NDArray,
    VELOCITY_MEASURE_VARIANCE: float,
    grid: DiscretizationGrid,
) -> KalmanResult:
    """
    Run the extended Kalman filter to estimate velocity from trajectory measurements

    Args:
        discretized_trajectories: dataframe of trajectories
        P_0: initial covariance matrix (n_cells x n_cells)
        v_0: initial velocity (n_cells x 1)
        flow: flow function
        Q: process noise covariance matrix (n_cells x n_cells)
        VELOCITY_MEASURE_VARIANCE: velocity measurement variance
        grid_params: discretization parameters (dx, dt, n_cells, n_timesteps)

    Returns:
        KalmanResult: 
            rho_hat: density (m^-1) (n_timesteps x n_cells) 
            v_hat: velocity (m/s) (n_timesteps x n_cells)
            std_rho: standard deviation of density (m^-1) (n_timesteps x n_cells)
            std_speed: standard deviation of speed (m/s) (n_timesteps x n_cells)
    """
    N_TIMESTEPS = grid.n_timesteps
    N_CELLS = grid.n_cells
    DX_METERS = grid.dx_meters
    DT_SECONDS = grid.dt_seconds

    # Init empty matrices to fill
    v_hat   = np.empty((N_TIMESTEPS, N_CELLS))
    v_preds   = np.empty((N_TIMESTEPS, N_CELLS))
    rho_hat = np.empty((N_TIMESTEPS, N_CELLS))
    std_speed = np.empty((N_TIMESTEPS, N_CELLS))
    std_rho = np.empty((N_TIMESTEPS, N_CELLS))
    cov_pred = np.empty((N_TIMESTEPS, N_CELLS, N_CELLS))
    cov = np.empty((N_TIMESTEPS, N_CELLS, N_CELLS))
    jacobian = np.empty((N_TIMESTEPS, N_CELLS, N_CELLS))

    v_hat[0, :] = v_0
    v_preds[0, :] = v_0
    P           = P_0
    rho_hat[0, :] = flow.density_from_velocity(v_0)
    std_speed[0] = np.sqrt(np.diag(P))
    std_rho[0] = std_speed[0] * flow.drho_dv(v_hat[0])
    cov_pred[0] = P_0
    cov[0] = P_0

    for i in range(N_TIMESTEPS - 1):
        rho_pred = godunov_step(
            rho_hat[i], Q=flow, dt=DT_SECONDS, dx=DX_METERS
        )  # non linear state update
        F = godunov_jacobian(rho_pred, Q=flow, dt=DT_SECONDS, dx=DX_METERS, with_params=False)
        H, y_v, R = get_measure_at_timestep(
            discretized_trajectories.filter(pl.col.t_index.is_between(i, i+1)),
            N_CELLS,
            VELOCITY_MEASURE_VARIANCE
        )

        # Change of variable from velocity to density
        v_pred = flow.velocity_from_density(rho_pred)
        # We have v_{i+1} = f(v_i) = V ° Godunov ° Rho(v_i)
        # For greenshields, rho and v are related linearly so the jacobian stays the same
        # For other flows, the chain rule may give different results
        # The functions are mutual inverses but they aren't evaluated at the same point
        F = np.diag(flow.dv_drho(rho_pred)) @ F @ np.diag(flow.drho_dv(v_hat[i]))
        y_pred = H @ v_pred
        v_corrected, _ , P = extended_kalman_step(v_pred, y_v, P, F, y_pred, H, Q, R)
        v_corrected = np.clip(v_corrected, 0, flow.v_max)

        v_hat[i+1] = v_corrected
        v_preds[i+1] = v_pred
        rho_hat[i + 1] = flow.density_from_velocity(v_corrected)
        std_speed[i + 1] = np.sqrt(np.diag(P))
        std_rho[i + 1] = std_speed[i + 1] * flow.drho_dv(v_corrected)
        cov_pred[i+1] = F @ P @ F.T + Q
        cov[i+1] = P
        jacobian[i] = F

    return KalmanResult(
        rho_hat=rho_hat,
        v_pred=v_preds,
        v_hat=v_hat,
        std_rho=std_rho,
        std_speed=std_speed,
        cov_pred=cov_pred,
        cov=cov,
        jacobian=jacobian
    )

def faster_predict(enkf):
    """ Predict next position using batch operations. """

    N = enkf.N
    # for i, s in enumerate(self.sigmas): # remove for loop
    enkf.sigmas = enkf.fx(enkf.sigmas, enkf.dt)

    e = multivariate_normal(enkf._mean, enkf.Q, N)
    enkf.sigmas += e

    enkf.x = np.mean(enkf.sigmas, axis=0)
    enkf.P = outer_product_sum(enkf.sigmas - enkf.x) / (N - 1)

    # save prior
    enkf.x_prior = np.copy(enkf.x)
    enkf.P_prior = np.copy(enkf.P)

def run_enkf_on_trajectories(
    discretized_trajectories: pl.DataFrame,
    P_0 : NDArray,
    v_0: NDArray | float,
    flow: GreenshieldsFlow|TriangularFlow|QuadraticLinearFlow,
    Q: NDArray,
    VELOCITY_MEASURE_VARIANCE: float,
    grid: DiscretizationGrid,
    n_particles: int = 100
) -> np.ndarray:
    """
    Run the ensemble Kalman filter to estimate velocity from trajectory measurements

    Args:
        discretized_trajectories: dataframe of trajectories
        P_0: initial covariance matrix (n_cells x n_cells)
        v_0: initial velocity (n_cells x 1)
        flow: flow function
        Q: process noise covariance matrix (n_cells x n_cells)
        VELOCITY_MEASURE_VARIANCE: velocity measurement variance
        grid_params: discretization parameters (dx, dt, n_cells, n_timesteps)

    Returns:
        v_hat: velocity (m/s) (n_timesteps x n_cells)
    """
    N_TIMESTEPS = grid.n_timesteps
    N_CELLS = grid.n_cells
    DX_METERS = grid.dx_meters
    DT_SECONDS = grid.dt_seconds

    # Init empty matrices to fill
    v_hat   = np.empty((N_TIMESTEPS, N_CELLS))

    def lwr_model(v,dt):
        rho = flow.density_from_velocity(v)
        rho_next = godunov_step(
            rho, Q=flow, dt=dt, dx=DX_METERS
        )  # non linear state update
        v_next = flow.velocity_from_density(rho_next)
        v_next = np.clip(v_next, 1e-3, flow.v_max)
        return v_next

    enKF = EnsembleKalmanFilter(
        x=v_0*np.ones(N_CELLS),
        P=P_0,
        dim_z=N_CELLS,  # static measurement size
        fx=lwr_model,  # non linear state update
        hx=lambda v: v,  # placeholder
        N=n_particles,
        dt=DT_SECONDS,
    )
    enKF.Q = Q
        
    v_hat[0, :] = v_0

    for i in range(N_TIMESTEPS - 1):
        H, y_v, R = get_measure_at_timestep(
            discretized_trajectories.filter(pl.col.t_index.is_between(i, i+1)),
            N_CELLS,
            VELOCITY_MEASURE_VARIANCE
        )
        if H.shape[0] == 0:
            # No measurements, just predict
            faster_predict(enKF)
            v_hat[i+1] = enKF.x
            continue
        enKF.R = R
        enKF.hx = lambda v: H @ v  # update measurement function with current H
        enKF._mean_z = np.zeros(H.shape[0])  # set mean_z to match measurement size
        enKF.update(y_v)
        v_hat[i+1] = enKF.x_post
        # enKF.predict()  # predict next state for the next iteration
        faster_predict(enKF)
    return v_hat


def run_ctm_enkf_on_trajectories(
    discretized_trajectories: pl.DataFrame,
    P_0 : NDArray,
    v_0: NDArray | float,
    flow: GreenshieldsFlow|TriangularFlow|QuadraticLinearFlow,
    Q: NDArray,
    VELOCITY_MEASURE_VARIANCE: float,
    grid: DiscretizationGrid,
    ramp_index: np.ndarray,
    n_particles: int = 100,
) -> np.ndarray:
    """
    Run the ensemble Kalman filter to estimate velocity from trajectory measurements

    Args:
        discretized_trajectories: dataframe of trajectories
        P_0: initial covariance matrix (n_cells x n_cells)
        v_0: initial velocity (n_cells x 1)
        flow: flow function
        Q: process noise covariance matrix (n_cells x n_cells)
        VELOCITY_MEASURE_VARIANCE: velocity measurement variance
        grid_params: discretization parameters (dx, dt, n_cells, n_timesteps)

    Returns:
        v_hat: velocity (m/s) (n_timesteps x n_cells)
    """
    N_TIMESTEPS = grid.n_timesteps
    N_CELLS = grid.n_cells
    DX_METERS = grid.dx_meters
    DT_SECONDS = grid.dt_seconds

    # Init empty matrices to fill
    v_hat   = np.empty((N_TIMESTEPS, N_CELLS))

    def ctm_model(u,dt):
        v = u[...,:N_CELLS]
        sources = u[...,N_CELLS:]
        rho = flow.density_from_velocity(v)
        rho_next = godunov_step(
            rho, Q=flow, dt=dt, dx=DX_METERS
        )  # non linear state update
        rho_next[...,ramp_index] += sources * dt / DX_METERS # add source term
        v_next = flow.velocity_from_density(rho_next)
        v_next = np.clip(v_next, 1e-3, flow.v_max)
        return np.concatenate((v_next, sources), axis=-1)  # state is velocity + source term

    enKF = EnsembleKalmanFilter(
        x=np.concatenate([v_0*np.ones(N_CELLS), np.zeros(len(ramp_index))]),  # state is velocity + source term
        P=P_0,
        dim_z=N_CELLS,  # placeholder
        fx=ctm_model,  # non linear state update
        hx=lambda v: v,  # placeholder
        N=n_particles,
        dt=DT_SECONDS,
    )
    enKF.Q = Q
    v_hat[0, :] = v_0

    for i in range(N_TIMESTEPS - 1):
        H, y_v, R = get_measure_at_timestep(
            discretized_trajectories.filter(pl.col.t_index.is_between(i, i+1)),
            N_CELLS,
            VELOCITY_MEASURE_VARIANCE
        )
        if H.shape[0] == 0:
            # No measurements, just predict
            faster_predict(enKF)
            v_hat[i+1] = enKF.x[...,:N_CELLS]
            continue
        enKF.R = R
        enKF.hx = lambda u: H @ u[...,:N_CELLS]  # update measurement function with current H
        enKF._mean_z = np.zeros(H.shape[0])  # set mean_z to match measurement size
        enKF.update(y_v)
        v_hat[i+1] = enKF.x_post[...,:N_CELLS]
        # enKF.predict()  # predict next state for the next iteration
        faster_predict(enKF)
    return v_hat


def run_rts_smoother(
        fwd: KalmanResult,
        Q: NDArray,
        flow: GreenshieldsFlow|TriangularFlow|QuadraticLinearFlow,

):
    """
    Run backward RTS smoothing
    
    args:
    fwd: Forward Kalman result

    returns v_smooth, P_smooth
    """
    N_TIMESTEPS = fwd.v_hat.shape[0]
    v_smooth = np.empty_like(fwd.v_hat)
    P_smooth = np.empty_like(fwd.cov_pred)
    v_smooth[-1] = fwd.v_hat[-1]
    P_smooth[-1] = fwd.cov[-1]
    for k in range(N_TIMESTEPS-2,-1,-1):
        v_smooth[k], P_smooth[k] = rts_smoother_step(
            x_k=        fwd.v_hat[k],
            x_kp1_pred= fwd.v_pred[k+1],
            xs_kp1=     v_smooth[k+1],
            P_k=        fwd.cov[k],
            P_kp1_pred= fwd.cov_pred[k+1],
            Ps_kp1=     P_smooth[k+1],
            F_k=        fwd.jacobian[k],
            Q=          Q,
        )
        v_smooth[k] = np.clip(v_smooth[k], 0, flow.v_max)
        P_smooth[k] = np.clip(P_smooth[k], -flow.v_max**2, flow.v_max**2)
    return v_smooth, P_smooth


def rts_smoother_step(
    x_k: NDArray,
    x_kp1_pred: NDArray,
    xs_kp1: NDArray,
    P_k: NDArray,
    P_kp1_pred: NDArray,
    Ps_kp1: NDArray,
    F_k: NDArray,
    Q: NDArray,
):
    """
    x_k: state estimate x_k|k
    x_kp1_pred: predicted x_k+1|k = f(x_k|k)
    xs_kp1: smoothed state estimate x_k+1|N
    P_k: covariance estimate P_k|k
    Ps_kp1: smoothed covariance estimate P_k+1|N
    P_kp1_pred: predicted covariance estimate P_k+1|k
    F_k: state transition jacobian df/dx at x_k|k
    Q: process noise covariance

    returns:
    xs_k: smoothed state estimate x_k|N
    Ps_k: smoothed covariance estimate P_k|N
    """
    S = P_kp1_pred

    # Method 1: direct inverse
    # G = P_k @ F_k.T @ np.linalg.inv(S)

    # Method 2: cholesky
    L = np.linalg.cholesky(S)  # Cholesky: S = L @ L.T
    # # G = P F.T (L L.T)^-1
    # # <=> G.T = (L L.T)^-1 F P
    # # <=> L L.T G.T = F P
    # # Solve n triangular systems 
    LtGt = scipy.linalg.solve_triangular(L, F_k @ P_k, lower=True)  # L (L.T G.T) = F @ P
    G = scipy.linalg.solve_triangular(L.T, LtGt).T  # L.T G.T = LtKt

    xs_k = x_k + G @ (xs_kp1 - x_kp1_pred)
    Ps_k = P_k + G @ (Ps_kp1 - P_kp1_pred) @ G.T
    # Joseph form for stability #https://ipnpr.jpl.nasa.gov/progress_report/42-233/42-233A.pdf
    I = np.eye(P_k.shape[0])
    Ps_k = (I - G @ F_k) @ P_k @ (I - G @ F_k).T + G @ (Ps_kp1 + Q) @ G.T

    return xs_k, Ps_k
