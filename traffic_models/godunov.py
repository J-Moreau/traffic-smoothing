# This file contains code that implements a baseline Godunov solver for LWR-Greenshield traffic model
# using initial code from P.Goatin, June 2017
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from traffic_models.flows import FlowFunction, GreenshieldsFlow, QuadraticLinearFlow


def godunov_step(
    rho: NDArray,
    Q: FlowFunction,
    dt: float,
    dx: float,
    left_boundary_Q: Optional[NDArray] = None,
    right_boundary_Q: Optional[NDArray] = None,
) -> NDArray:
    """
    Make one prediction with Godunov scheme from initial density data rho
    supports batched arrays of shape (batch_size, n_cells) or (n_cells,)

    Godunov numerical flux for a CONCAVE flux f with max value at rho = rho_c

    Args:
        rho: Initial cell densities rho_t
        flow
        dt
        dx
        left_boundary_Q: boundary condition on the inflow
        right_boundary_Q: boundary condition on the outflow

    Returns:
        rho_{t+1}
    """
    rho_c = Q.rho_c

    demand = Q(rho) * (rho < rho_c) + Q(rho_c) * (rho >= rho_c)
    supply = Q(rho) * (rho > rho_c) + Q(rho_c) * (rho <= rho_c)
    supply_ip1 = np.roll(supply, -1, axis=-1)  # left shift
    if right_boundary_Q is None:
        # If there is no boundary condition
        # We assume that the supply is constant at the right boundary
        # supply_{N+1} <- supply_{N} # mirror boundary
        supply_ip1[..., -1] = supply_ip1[..., -2]  # mirror boundary condition
    else:
        supply_ip1[..., -1] = right_boundary_Q
    # Godunov output flux (at the right boundary)
    # Q_{i+1/2} = min(demand_i, supply_{i+1})
    Q_right = np.minimum(demand, supply_ip1)
    # input flux (at the left boundary) i.e. Q_{i-1/2}
    Q_left = np.roll(Q_right, 1, axis=-1)  # right shift

    # Again, if there is no boundary condition
    # We assume that the incoming demand is constant:
    # here this reads as
    # demand_{-1} <- demand_0 # mirror boundary
    # Q_{-1/2} <- min(demand_{-1},supply_{0}) # flux formula
    if left_boundary_Q is None:
        left_boundary_flow = demand[..., 0]  # mirror boundary condition
    else:
        left_boundary_flow = left_boundary_Q
    Q_left[..., 0] = np.minimum(left_boundary_flow, supply[..., 0])

    rho_next = rho + (dt / dx) * (Q_left - Q_right)
    return rho_next


def godunov_jacobian(
    rho: NDArray,
    Q: GreenshieldsFlow | QuadraticLinearFlow,
    dt: float,
    dx: float,
    with_params: bool = False,
) -> NDArray:
    """
    Godunov jacobian of numerical flux for a CONCAVE flow Q with max value at rho = rho_c

    Args:
        rho: Initial cell densities rho_t
        flow
        d_flow
        dt
        dx

    Returns:
        rho_{t+1}
    """
    rho_c = Q.rho_c

    demand_i = Q(rho) * (rho < rho_c) + Q(rho_c) * (rho >= rho_c)
    supply_i = Q(rho) * (rho > rho_c) + Q(rho_c) * (rho <= rho_c)

    supply_ip1 = np.roll(supply_i, -1)  # supply_i+1 left shift
    supply_ip1[-1] = supply_ip1[-2]  # mirror boundary condition

    demand_im1 = np.roll(demand_i, 1)  # right shift
    demand_im1[0] = demand_im1[1]  # mirror boundary condition

    d_demand_i = Q.derivative(rho) * (rho < rho_c) * (demand_i < supply_ip1)
    d_supply_i = Q.derivative(rho) * (rho > rho_c) * (demand_im1 > supply_i)

    # (d_Q_right_i/ d_rho_j != 0) => (j == i+1)
    # so d_supply_i+1 / d_rho_j is an upper diagonal
    # output j depends on i+1
    # Q_right = np.minimum(demand_i, supply_ip1)

    # Godunov flux (at the right boundary)
    # the right interface is diagonal + upper diagonal
    dQ_right = np.diag(d_demand_i) + np.diag(d_supply_i[1:], k=1)

    # the left interface is diagonal + lower diagonal
    dQ_left = np.diag(d_demand_i[:-1], k=-1) + np.diag(d_supply_i)

    # rho_next = G(rho) = rho + (dt / dx) * (Q_left - Q_right)
    d_G_d_rho = np.eye(rho.shape[0]) + (dt / dx) * (dQ_left - dQ_right)

    if not with_params:
        return d_G_d_rho

    ### parameter derivatives

    rho_im1 = np.roll(rho, 1)
    rho_im1[0] = rho_im1[1]  # mirror boundary condition
    dQdp_left = (
        Q.dflow_dparam(np.minimum(rho, rho_c)) * (demand_im1 > supply_i)[:, np.newaxis]
        + Q.dflow_dparam(np.maximum(rho_im1, rho_c))
        * (demand_im1 <= supply_i)[:, np.newaxis]
    )
    dQdp_right = np.roll(dQdp_left, -1)  # left shift
    dQdp_right[-1] = dQdp_right[-2]  # mirror boundary condition

    d_G_d_param = (dt / dx) * (dQdp_left - dQdp_right)
    # returns an array of shape (n+p, n+p)
    return np.block(
        [
            [d_G_d_rho, d_G_d_param],
            [np.zeros_like(d_G_d_param).T, np.eye(d_G_d_param.shape[1])],
        ]
    )
