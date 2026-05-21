# This file contains code that implements a baseline Godunov solver for LWR-Greenshield traffic model
# using initial code from P.Goatin, June 2017
from typing import Callable

import torch
import torch.linalg

from traffic_models.nn.flows_torch import ConvexVelocityFlow, GreenshieldsFlow


def godunov_step(
    rho: torch.Tensor,
    Q: Callable[[torch.Tensor], torch.Tensor],
    rho_c: torch.Tensor,
    dt: float,
    dx: float,
    detach_boundaries: bool = False,
) -> torch.Tensor:
    """
    Make one prediction with Godunov scheme from initial density data rho using torch tensors.
    (for concave flow)
    the last dimension is expected to be the space dimension
    e.g. this function works with
    rho.shape = (n_cells,)
    rho.shape = (n_timesteps, n_cells)
    rho.shape = (n_batchs, n_timesteps, n_cells)

    Args:
        rho: Initial cell densities (torch tensor)
        Q: Flow function (callable on torch tensors)
        rho_c: Flow function critical rho
        dt: Time step
        dx: Cell length
        detach_boundaries: whether to detach the boundary conditions from the computational graph
        useful for computing jacobians

    Returns:
        rho_{t+1} (torch tensor)
    """
    # see the numpy implementation for explanations
    demand = torch.where(rho < rho_c, Q(rho), Q(rho_c))
    supply = torch.where(rho > rho_c, Q(rho), Q(rho_c))

    supply_ip1 = torch.roll(supply, shifts=-1, dims=-1)
    right_boundary = supply_ip1[..., -2].detach() if detach_boundaries else supply_ip1[..., -2]
    supply_ip1[..., -1] = right_boundary

    Q_out = torch.minimum(demand, supply_ip1)

    Q_in = torch.roll(Q_out, shifts=1, dims=-1)
    left_boundary = demand[..., 0].detach() if detach_boundaries else demand[..., 0]
    Q_in[..., 0] = torch.minimum(left_boundary, supply[..., 0])

    rho_next = rho + (dt / dx) * (Q_in - Q_out)
    return rho_next


def convex_godunov_step(
    v: torch.Tensor,
    R: Callable[[torch.Tensor], torch.Tensor],
    v_c: torch.Tensor,
    dt: float,
    dx: float,
) -> torch.Tensor:
    """
    Make one prediction with Godunov scheme for convex flow R(v)
    from velocity data v using torch tensors.
    v_t + R(v)_x = 0

    Args:
        v: Initial cell velocities (torch tensor)
        Q: Flow function of velocity
        v_c: Flow function critical v
        dt: Time step
        dx: Cell length

    Returns:
        v_{t+1} (torch tensor)
    """
    demand = torch.where(v < v_c, R(v_c), R(v))
    supply = torch.where(v > v_c, R(v_c), R(v))

    supply = torch.roll(supply, shifts=-1, dims=0)
    supply[-1] = supply[-2].detach()  # detach boundary condition

    Q_out = torch.maximum(demand, supply)

    Q_in = torch.roll(Q_out, shifts=1, dims=0)
    Q_in[0] = torch.maximum(demand[0].detach(), supply[0])  # detach boundary condition

    v_next = v + (dt / dx) * (Q_in - Q_out)
    return v_next


def extended_kalman_step(
    x_pred: torch.Tensor,
    y: torch.Tensor,
    P: torch.Tensor,
    F: torch.Tensor,
    H: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extended Kalman filter for torch tensors, using Cholesky decomposition for stability.

    Args:
        x_pred: next state estimate x_{k+1|k} (n x 1)
        y: measurement y_{k+1} (m x 1)
        P: covariance estimate (n x n)
        F: state transition jacobian (n x n)
        H: measurement matrix (m x n)
        Q: process noise covariance (n x n)
        R: measurement noise covariance (m x m)

    Returns:
        x_corrected: updated state estimate
        y_pred: measurement estimate
        P_pred: covariance estimate
    """
    # Predict
    P = F @ P @ F.T + Q

    # Update
    S = H @ P @ H.T + R  # Innovation covariance

    # Cholesky decomposition
    L = torch.linalg.cholesky(S)
    u = torch.linalg.solve_triangular(L, H @ P, upper=False)
    K = torch.linalg.solve_triangular(L.T, u, upper=True).T

    y_pred = H @ x_pred
    x_corrected = x_pred + K @ (y - y_pred)
    Id = torch.eye(x_pred.shape[0], dtype=x_pred.dtype, device=x_pred.device)
    P_pred = (Id - K @ H) @ P
    return x_corrected, y_pred, P_pred


def godunov_jacobian(
    rho: torch.Tensor, flow: GreenshieldsFlow, dt: float, dx: float
) -> torch.Tensor:
    return torch.autograd.functional.jacobian(
        lambda rho: godunov_step(
            rho, flow, torch.exp(flow.get_parameter("log_rho_max")) / 2, dt, dx
        ),
        rho,
        create_graph=True,
    )
