import numpy as np
import polars as pl

from traffic_models.flows import TriangularFlow


def kalman_step(
    x: np.ndarray,
    y: np.ndarray,
    P: np.matrix,
    F: np.matrix,
    H: np.matrix,
    Q: np.matrix,
    R: np.matrix,
) -> tuple[np.ndarray, np.ndarray, np.matrix]:
    """
    General Kalman filter using Cholesky decomposition for stability.

    Parameters:
        x:   state estimate (n x 1)
        y:   measurement (m x 1)
        P:   covariance estimate (n x n)
        F:   state transition matrix (n x n)
        H:   measurement matrix (m x n)
        Q:   process noise covariance (n x n)
        R:   measurement noise covariance (m x m)

    Returns:
        x_pred: state estimate
        y_pred: measurement estimate
        P_pred: covariance estimate
    """

    # === Predict ===
    x = F @ x
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

    # Method 3: Cholesky
    L = np.linalg.cholesky(S)  # Cholesky: S = L @ L.T
    # # Solve for K using forward and backward substitution
    # # L L^T K.T = H P
    u = np.linalg.solve(L, H @ P)  # L (L.T K.T) = H @ P
    K = np.linalg.solve(L.T, u).T  # L.T K.T = u

    # Method 4: Thomas algorithm (for k-diagonal matrices)
    # TODO
    
    y_pred = H @ x  # Innovation
    x_pred = x + K @ (y - y_pred)  # State update
    P_pred = (np.eye(x.shape[0]) - K @ H) @ P  # Covariance update
    return x_pred, y_pred, P_pred


def get_measure_at_timestep(
    trajectories_at_t: pl.DataFrame,
    N_CELLS: int,
    velocity_variance: float,
    is_relative_var: bool = False,
    corrected: bool = False,
    # flow:FlowFunction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read measured velocity at timestep t from a table of discretized trajectories
    For use in kalman filter, returns additional matrices H and R where y = Hx + R^{1/2} e
    y is the measure, x is the state and e ~ N(0,I)

    Use is_relative_var=True to scale the variance with the mean of the measure values:
    * additive errors: velocity_variance=var(Y), is_relative_var=False
    * multiplicative errors: velocity_variance=var(Y/mean(Y)), is_relative_var=True

    Returns linear measurement matrix H_t, measure values y_t and covariance matrix R_t all in terms of velocity
    """
    flow = TriangularFlow(rho_c=0.0186, Q_c=0.56, rho_max=0.127)

    measure = (
        trajectories_at_t
        .group_by("x_index")
        .agg(pl.col("velocity").mean(), n_measures_in_cell=pl.len())
        .sort("x_index")
    )
    corrected_measure = (
        trajectories_at_t
        .with_columns(
            density = flow.density_from_velocity(trajectories_at_t["velocity"].to_numpy()),
        )
        .group_by("x_index")
        .agg(velocity=(pl.col.velocity**2/pl.col.density).sum()/(pl.col.velocity/pl.col.density).sum(), n_measures_in_cell=pl.len())
        .sort("x_index")
    )
    if corrected:
        measure = corrected_measure
    measure_x_indexes = measure["x_index"].to_numpy()
    measure_values = measure["velocity"].to_numpy()
    n_measures_in_cell = measure["n_measures_in_cell"].to_numpy()
    n_measures = measure_x_indexes.shape[0]
    H = np.zeros((n_measures, N_CELLS))
    H[np.arange(n_measures), measure_x_indexes] = 1  # / n_measures_in_cell
    y = measure_values
    # variance of the mean of n variables with variance sigma^2 is sigma^2/n
    R = np.diag(1 / n_measures_in_cell) * velocity_variance
    # if we're using relative variance var(X/mean(X))
    if is_relative_var:
        R *= measure_values**2
    return H, y, R

