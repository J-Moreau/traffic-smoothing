import numpy as np
import polars as pl

from traffic_models.data.discretize import discretize_trajectories
from traffic_models.data.mobile_century import read_mobile_century_data
from traffic_models.data.ngsim import load_ngsim_data
from traffic_models.data.trajectory import (
    add_relative_time,
    clip_trajectories,
    downsample_trajectories,
)
from traffic_models.dense_fields import FlowData, agg_trajectories_into_field
from traffic_models.experiment.probe_experiment import ProbeExperimentConfig
from traffic_models.sim import DiscretizationGrid


def prepare_probe_experiment_data(
    conf: ProbeExperimentConfig,
) -> tuple[DiscretizationGrid, pl.DataFrame, pl.DataFrame, FlowData]:
    if conf.data.name == "mobile-century":
        if conf.data.xmin_meters is None or conf.data.xmax_meters is None:
            args = ()
        else:
            args = (conf.data.xmin_meters, conf.data.xmax_meters)
        trajectories = read_mobile_century_data(conf.data.path, *args)

    elif conf.data.name in ("us-101", "i-80"):
        trajectories = load_ngsim_data(
            path=conf.data.path, location=conf.data.name, lanes=conf.data.lanes
        )
    else:
        raise ValueError(f"Unknown dataset name: {conf.data.name}")

    trajectories = trajectories.filter(
        (pl.col.time_seconds > conf.data.start_seconds)
        & (pl.col.time_seconds < conf.data.end_seconds)
    ).pipe(add_relative_time)

    assert trajectories.shape[0] > 0, "Trajectories are empty"
    tmax: float = trajectories["time_seconds"].max()  # type: ignore
    xmax: float = trajectories["x_meters"].max()  # type: ignore
    trajectories = trajectories.pipe(
        clip_trajectories, xmin=0, xmax=xmax, start_seconds=0, end_seconds=tmax
    )  # delete the points at the boundary of the domain (t=tmax or x=xmax)

    if conf.data.name in ("us-101", "i-80", "mobile-century", "a86"):
        fields = agg_trajectories_into_field(
            trajectories=trajectories,
            xmax=xmax,
            tmax=tmax,
            space_bin_meters=conf.dx_meters,
            time_bin_seconds=conf.dt_seconds,
            smoothing=conf.data.smoothing,
        )
        # assert np.quantile(fields.speed,0.90) <= conf.dx_meters / conf.dt_seconds, "CFL condition not met"
        # CFL condition: (Vmax * dt) / dx < 1
    else:
        fields = None

    grid = DiscretizationGrid.from_dimensions(
        dx_meters=conf.dx_meters, dt_seconds=conf.dt_seconds, lmax=xmax, tmax=tmax
    )

    trajectories_with_grid = trajectories.with_columns(
        t_index=(pl.col.time_seconds / grid.dt_seconds).floor().cast(pl.Int64),
        x_index=(pl.col.x_meters / grid.dx_meters).floor().cast(pl.Int64),
    )
    # Select a larger sample (20%) of data points in the first and last x_index
    boundary_measures = (
        trajectories
        # .pipe(add_relative_time)
        .pipe(
            downsample_trajectories,
            fraction=conf.data.boundary_fraction,
            seed=conf.seed,
        )
        .pipe(
            discretize_trajectories,
            grid.dt_seconds,
            grid.dx_meters,
            grid.n_cells,
            grid.n_timesteps,
        )
        .filter((pl.col.x_index == 0) | (pl.col.x_index == grid.n_cells - 1))
        .with_columns(
            pl.col.vehicle_id + "_boundary", is_boundary_information=pl.lit(True)
        )
    )
    if conf.data.probe_fraction == 0:
        return grid, boundary_measures, trajectories_with_grid, fields

    # Downsample to get probe vehicles
    sample_trajectories = downsample_trajectories(
        trajectories, fraction=conf.data.probe_fraction, seed=conf.seed
    )

    discretized_trajectories = discretize_trajectories(
        sample_trajectories,
        grid.dt_seconds,
        grid.dx_meters,
        grid.n_cells,
        grid.n_timesteps,
    )
    discretized_trajectories = pl.concat(
        [
            discretized_trajectories.with_columns(
                is_boundary_information=pl.lit(False)
            ),
            boundary_measures,
        ]
    ).unique(("vehicle_id", "t_index"))
    return grid, discretized_trajectories, trajectories_with_grid, fields
