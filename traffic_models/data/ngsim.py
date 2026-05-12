from functools import cache
from typing import Literal

import numpy as np
import polars as pl

from traffic_models.data.trajectory import add_relative_time, create_unique_vehicle_id

SAMPLE_INTERVAL_MILLISECONDS = 100  # this is a fixed value from the dataset
FEET_PER_METER = 3.28084


def load_trajectories(
    table_path: str,
    road_location: Literal["i-80", "us-101"],
    lanes: set[int] = {3, 4, 5},
) -> pl.DataFrame:
    
    if table_path.endswith(".csv"):
        raw_trajectories: pl.DataFrame = pl.read_csv(table_path)
    elif table_path.endswith(".parquet"):
        raw_trajectories: pl.DataFrame = pl.read_parquet(table_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
    return (
        raw_trajectories.unique()  # remove some duplicates in data around 8:00:00
        .rename({k: k.lower() for k in raw_trajectories.columns})
        .rename(dict(v_vel="velocity", local_y="x_meters", local_x="y_meters"))
        .with_columns(
            pl.col.global_time.cast(pl.Datetime("ms", time_zone="US/Pacific"))
        )
        .with_columns(
            pl.col.velocity / FEET_PER_METER,
            pl.col.x_meters / FEET_PER_METER,
            pl.col.space_headway / FEET_PER_METER,
        )
        .filter(pl.col.location == road_location)
        .filter(pl.col.lane_id.is_in(lanes))
        .with_columns(pl.col.vehicle_id.cast(pl.String))
    )

def load_ngsim_data(
    path: str = "data/ngsim/NGSIM_trajectories.parquet",
    lanes: tuple[int,...] = (1, 2, 3, 4, 5, 6, 7, 8),  # all lanes
    location="us-101",
) -> pl.DataFrame:
    """
    Returns trajectories and a grid of aggregated field quantities from all vehicles.

    start_seconds and end_seconds are relative to the start of the dataset.

    grid parameters:
    - `len_x_grid`: number of bins in the x direction (space)
    - `time_bin_seconds`: size of time bins in seconds

    Common values for the `lanes` argument
    ```
    lanes = {3,4,5} # just the middle (easier)
    lanes = {1,2,3,4,5,6,7} # no in-ramp
    lanes = {1, 2, 3, 4, 5, 6, 7, 8} # all lanes
    ```
    """
    # these are the approximate start and end times of the dataset
    # start_time = datetime(2005, 6, 15, 7, 50, 0)
    # end_time = datetime(2005, 6, 15, 8, 35, 0)
    # further restrict to a time window (relative to start_time)
    lanes = set(lanes)
    N_LANES = len(lanes)

    trajectories = (
        load_trajectories(path, location, lanes=lanes)
        .pipe(add_relative_time)
        .pipe(create_unique_vehicle_id)
    )
    # Clip trajectories
    # xmin: float = trajectories["x_meters"].quantile(0.)  # type:ignore
    # xmax: float = trajectories["x_meters"].quantile(1.)  # type:ignore
    if location == "us-101":
        xmin = 20
        xmax = 620
    elif location == "i-80":
        xmin = 30
        xmax = 480

    return trajectories.filter(pl.col.x_meters > xmin).filter(
            pl.col.x_meters < xmax
        ).with_columns(x_meters=pl.col.x_meters - xmin)
