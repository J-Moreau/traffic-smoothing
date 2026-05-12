import numpy as np
import polars as pl

from traffic_models.metrics import trajectory_mse


def test_trajectory_mse():
    # Create a simple trajectories DataFrame with three measurements
    trajectories = pl.DataFrame({
        "t_index": [0, 0, 1],
        "x_index": [0, 1, 0],
        "velocity": [10.0, 20.0, 30.0],
    })

    # v_pred shaped (n_timesteps, n_cells)
    v_pred = np.array([
        [11.0, 19.0],  # timestep 0
        [29.0, 31.0],  # timestep 1
    ])

    # All squared errors are 1, mean should be 1.0
    mse = trajectory_mse(trajectories, v_pred)
    assert np.isclose(mse, 1.0)
