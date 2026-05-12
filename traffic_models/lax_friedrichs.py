from typing import Literal

import numpy as np
from numpy.typing import NDArray

from traffic_models.flows import ARZFlow, GreenshieldsFlow, TriangularFlow
from traffic_models.sim import DiscretizationGrid


def lax_friedrichs_affine_update(
    t_flow: TriangularFlow, rho: NDArray, grid: DiscretizationGrid
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Lax-Friedrichs jacobian matrix and bias term for a triangular flow.
    WITH RHO_C = RHO_MAX / 2
    $\rho_{t+1} = F \rho_t + B$
    """
    V_k = np.sign(t_flow.rho_max / 2 - rho) * t_flow.v_max

    F = (
        # 1/2 (\rho_{k+1} + \rho_{k-1})
        # - dt/2dx (Q(\rho_{k+1}) - Q(\rho_{k-1}))
        1 / 2 * (np.eye(grid.n_cells, k=1) + np.eye(grid.n_cells, k=-1))
        - grid.dt_seconds
        / (2 * grid.dx_meters)
        * (np.diag(V_k[1:], k=1) - np.diag(V_k[:-1], k=-1))
    )
    # At the boundaries, we assume that the value is constant (here it is given)
    F[0] = np.eye(grid.n_cells)[0, :]
    F[-1] = np.eye(grid.n_cells)[-1, :]

    # The linear component actually lacks an affine term dt/2dx * \rho_max V_max
    # when rho > rho_max/2
    B = (
        grid.dt_seconds
        / (2 * grid.dx_meters)
        * t_flow.rho_max
        * (np.roll(V_k.clip(max=0), -1) - np.roll(V_k.clip(max=0), 1))
    )
    B[0] = 0
    B[-1] = 0
    return F, B


def lax_friedrichs_step(
    rho: NDArray, flow: TriangularFlow | GreenshieldsFlow, grid: DiscretizationGrid
) -> np.ndarray:
    # 1/2 (\rho_{k+1} + \rho_{k-1})
    # - dt/2dx (Q(\rho_{k+1}) - Q(\rho_{k-1}))
    rho_next = 1 / 2 * (np.roll(rho, 1) + np.roll(rho, -1)) - grid.dt_seconds / (
        2 * grid.dx_meters
    ) * (flow(np.roll(rho, -1)) - flow(np.roll(rho, 1)))
    # at the boundaries we assume mirror rho values for ghost cells
    rho_next[0] = 1 / 2 * (rho [1] + rho[0]) - grid.dt_seconds / (
        2 * grid.dx_meters
    ) * (flow(rho[1]) - flow(rho[0]))
    rho_next[-1] = 1 / 2 * (rho [-2] + rho[-1]) - grid.dt_seconds / (
        2 * grid.dx_meters
    ) * (flow(rho[-1]) - flow(rho[-2]))
    return rho_next


def lax_friedrichs_step_batch(
    U: np.ndarray, flow: ARZFlow, grid: DiscretizationGrid
) -> np.ndarray:
    # U shape: (..., L) where L=space
    
    f_U = flow(U)
    U_pad = np.pad(U, (*([(0,0)]*(U.ndim-1)),(1,1)), mode="edge") # pad U with edge values for ghost cells
    f_U_pad = np.pad(f_U, (*([(0,0)]*(f_U.ndim-1)),(1,1)), mode="edge") # pad f(U) with edge values for ghost cells
    alpha = grid.dt_seconds / (2 * grid.dx_meters)
    U_next = 0.5 * (np.roll(U_pad, 1, axis=-1) + np.roll(U_pad, -1, axis=-1)) - alpha * (
        np.roll(f_U_pad, -1, axis=-1) - np.roll(f_U_pad, 1, axis=-1)
    )
    # mirror boundary conditions
    # U_next[..., 0] = 0.5 * (U[..., 0] + U[..., 1]) - alpha * (f_U[..., 1] - f_U[..., 0])
    # U_next[..., -1] = 0.5 * (U[..., -2] + U[..., -1]) - alpha * (f_U[..., -1] - f_U[..., -2])
    
    return U_next[...,1:-1] + grid.dt_seconds * flow.source_term(U)


def rusanov_step_batch(
    U: np.ndarray, flow: ARZFlow, grid: DiscretizationGrid, boundary: Literal["edge","wrap"]="edge"
) -> np.ndarray:
    """Rusanov scheme (also known as local Lax-Friedrichs) for second order traffic flow."""
    f_U = flow(U)
    speed = np.max(np.abs(flow.lambdas(U)), axis=-2) # from shape (..., 2, L) to (..., L)
    speed = np.pad(speed, (*([(0,0)]*(speed.ndim-1)),(1,1)), mode=boundary) # pad 
    alpha = np.maximum(speed[...,1:], speed[...,:-1])[..., np.newaxis, :] # from shape (..., L) to (..., 1, L)
    # alpha_i (i=0,n-1) = max(s_i, s_ip1)

    f_U = np.pad(f_U, (*([(0,0)]*(f_U.ndim-1)),(1,1)), mode=boundary) # pad
    U_pad = np.pad(U, (*([(0,0)]*(U.ndim-1)),(1,1)), mode=boundary) # pad
    U_i = U_pad[...,:-1]
    U_ip1 = U_pad[...,1:]
    f_U_i = f_U[...,:-1]
    f_U_ip1 = f_U[...,1:]
    
    F_iphalf = 1/2 * (f_U_i + f_U_ip1) - 0.5 * alpha * (U_ip1 - U_i)
    # F_iphalf (i=1/2,n-3/2) = 1/2 (f(U)_i + f(U)_ip1) - 1/2 alpha_i (U_ip1 - U_i)
    
    # F_imhalf (i=3/2,n-3/2) = F_iphalf[:-1]
    # F_iphalf (i=3/2,n-3/2) = F_iphalf[1:]
    U_next = U - grid.dt_seconds / grid.dx_meters * (F_iphalf[...,1:] - F_iphalf[...,:-1]) # F_iphalf - F_imhalf
    # mirror boundary conditions
    return U_next + grid.dt_seconds * flow.source_term(U)


def lax_friedrichs_rollout(
    U_0: NDArray,
    flow: ARZFlow,
    grid: DiscretizationGrid,
    solver: Literal["lax-friedrichs", "rusanov"]="lax-friedrichs"
) -> NDArray:
    solver_step = {
        "lax-friedrichs": lax_friedrichs_step_batch,
        "rusanov": rusanov_step_batch,
    }[solver]
    trajectory = np.zeros((grid.n_timesteps, 2, grid.n_cells))
    trajectory[0] = U_0
    U = U_0.copy()
    for t in range(1, grid.n_timesteps):
        U = solver_step(U, flow, grid)
        trajectory[t] = U
    return trajectory

