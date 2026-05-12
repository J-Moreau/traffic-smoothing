import numpy as np
import polars as pl

from traffic_models.flows import GreenshieldsFlow
from traffic_models.sim import DiscretizationGrid
from traffic_models.var_optim import FourDVarConfig, windowed_w4DVAR


def test_w4dvar_runs_and_shapes():
    # Dummy discretized trajectories
    n_cells = 4
    n_timesteps = 3
    data = {
        "vehicle_id": ["v1", "v1", "v1", "v2", "v2", ],
        "t_index": [0, 1, 2, 0, 1, ],
        "velocity": [5, 6, 7, 4, 5, ],
        "x_index": [0, 1, 2, 0, 1, ],
    }
    trajectories = pl.DataFrame(data)

    # Dummy grid
    grid = DiscretizationGrid(
        dx_meters=10,
        dt_seconds=1,
        n_cells=n_cells,
        n_timesteps=n_timesteps,
    )

    # Dummy flow
    flow = GreenshieldsFlow(v_max=10, rho_max=100)

    
    v_0 = np.ones(grid.n_cells)*1.0
    v_0[:2] = 0.0
    v_pred, _, _ = windowed_w4DVAR(
        trajectories=trajectories,
        v_0 = v_0,
        conf=FourDVarConfig(
            background_variance=1.0,
            model_variance=1.0,
            measurement_variance=1.0,
            n_iters=3,
            solver="godunov",    
        ),
        grid=grid,
        flow=flow,
        window_seconds=2,
    )

    assert v_pred.shape == (n_timesteps, n_cells)