import numpy as np
import polars as pl

from traffic_models.flows import GreenshieldsFlow
from traffic_models.naive_physics import (
    NaivePhysicsResult,
    naive_physics_on_trajectories,
)
from traffic_models.sim import DiscretizationGrid


def test_naive_physics_on_trajectories_runs_and_shapes():
    # Dummy discretized trajectories
    n_cells = 4
    n_timesteps = 3
    data = {
        "vehicle_id": ["v1", "v1", "v1", "v2", "v2"],
        "t_index": [0, 1, 2, 0, 1],
        "x_meters": [0, 10, 20, 5, 15],
        "velocity": [5, 6, 7, 4, 5],
        "x_index": [0, 1, 2, 0, 1],
    }
    df = pl.DataFrame(data)

    # Dummy grid
    grid = DiscretizationGrid(
        dx_meters=10,
        dt_seconds=1,
        n_cells=n_cells,
        n_timesteps=n_timesteps,
    )

    # Dummy flow
    flow = GreenshieldsFlow(v_max=10, rho_max=100)

    # Initial velocity
    v_0 = np.ones(n_cells) * 5

    # Run naive physics
    result = naive_physics_on_trajectories(
        discretized_trajectories=df,
        v_0=v_0,
        flow=flow,
        grid=grid,
        trust_ratio=1.0,
    )

    assert isinstance(result, NaivePhysicsResult)
    assert result.v_hat.shape == (n_timesteps, n_cells)
    assert result.rho_hat.shape == (n_timesteps, n_cells)
    # The filter uses a ±2 timestep window, so t_index 0..2 rows are all included at i=0.
    # x_index=0: mean([5,4])=4.5, x_index=1: mean([6,5])=5.5, x_index=2: mean([7])=7.0, x_index=3: no data → 5.0
    np.testing.assert_array_equal(result.v_hat[0], [4.5, 5.5, 7.0, 5.0])
