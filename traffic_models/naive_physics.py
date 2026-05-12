from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from traffic_models.flows import (
    ARZFlow,
    GreenshieldsFlow,
    QuadraticLinearFlow,
    TriangularFlow,
)
from traffic_models.godunov import godunov_step
from traffic_models.lax_friedrichs import lax_friedrichs_step_batch, rusanov_step_batch
from traffic_models.sim import DiscretizationGrid


@dataclass
class NaivePhysicsResult:
    rho_hat: NDArray
    v_hat: NDArray

def naive_physics_on_trajectories(
    discretized_trajectories: pl.DataFrame,
    v_0: NDArray | float,
    flow: GreenshieldsFlow|TriangularFlow|QuadraticLinearFlow|ARZFlow,
    grid: DiscretizationGrid,
    trust_ratio: float = 0.5,
) -> NaivePhysicsResult:
    """
    Estimate velocity from trajectory measurements

    Args:
        discretized_trajectories: dataframe of trajectories
        v_0: initial velocity (n_cells x 1)
        flow: flow function
        grid_params: discretization parameters (dx, dt, n_cells, n_timesteps)
        trust_ratio: ratio of trust in measurements vs model prediction (0: only model, 1: only measurements)

    Returns:
        NaivePhysicsResult: 
            rho_hat: density (m^-1) (n_timesteps x n_cells) 
            v_hat: velocity (m/s) (n_timesteps x n_cells)
    """
    N_TIMESTEPS = grid.n_timesteps
    N_CELLS = grid.n_cells
    DX_METERS = grid.dx_meters
    DT_SECONDS = grid.dt_seconds

    # Init empty matrices to fill
    v_hat   = np.zeros((N_TIMESTEPS, N_CELLS))
    rho_hat = np.zeros((N_TIMESTEPS, N_CELLS))

    v_hat[0, :] = v_0
    if isinstance(flow, ARZFlow):
        rho_hat[0] = flow.rho_eq(v_0)
    for i in range(N_TIMESTEPS - 1):
        measure = (
            discretized_trajectories
            # .filter(t_index=i)
            .filter(pl.col.t_index.is_between(i-2, i+2)) # use measurements in a time window around the current time step to increase the number of measurements (and reduce noise)
            .group_by("x_index")
            .agg(pl.col("velocity").mean().clip(0, flow.v_max))
            .sort("x_index")
        )
        measure_x_indexes = measure["x_index"].to_numpy()
        measure_values = measure["velocity"].to_numpy()
        v_hat[i, measure_x_indexes] = (1-trust_ratio) * v_hat[i, measure_x_indexes] + trust_ratio * measure_values
        if trust_ratio <= 1:
            if isinstance(flow, ARZFlow):
                U = np.stack([rho_hat[i], flow.q(rho_hat[i], v_hat[i])], axis=0)
                # U_pred = lax_friedrichs_step_batch(U, flow, grid)
                U_pred = rusanov_step_batch(U, flow, grid)
                rho_hat[i+1] = np.clip(U_pred[0], 1e-2, flow.rho_max-1e-6)
                v_pred = flow.v(rho_hat[i+1], U_pred[1])
                v_pred = np.clip(v_pred, 1e-6, flow.v_max-1)
            else:
                rho_hat[i] = flow.density_from_velocity(v_hat[i])
                rho_pred = godunov_step(
                    rho_hat[i], Q=flow, dt=DT_SECONDS, dx=DX_METERS
                )  # non linear state update
                rho_pred = np.clip(rho_pred, 1e-6, flow.rho_max-1e-6)
                # Change of variable from velocity to density
                v_pred = flow.velocity_from_density(rho_pred)
            # We have v_{i+1} = f(v_i) = V ° Godunov ° Rho(v_i)
            v_hat[i+1] = v_pred
        else:
            v_hat[i+1] = v_hat[i]

    return NaivePhysicsResult(
        rho_hat=rho_hat,
        v_hat=v_hat,
    )
