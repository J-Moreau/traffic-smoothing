from pathlib import Path

import polars as pl

from traffic_models.data.trajectory import (
    add_relative_time,
    clip_trajectories,
    create_unique_vehicle_id,
)


def read_mobile_century_data(
        trajectories_dir: str = "data/mobilecentury/NB_veh_files",
        xmin: float = 33_000,
        xmax: float = 46_000,
    ) -> pl.DataFrame:
    """
    for northbound trajectories:
        xmax_meters=46_000,
        xmin_meters=33_000,
    for southbound trajectories:
        xmax_meters=-33_000,
        xmin_meters=-45_000,
    """
    csv_dir = Path(trajectories_dir)
    files = sorted(csv_dir.glob("*.csv"))

    dfs = []
    for f in files:
        df = pl.read_csv(f).with_columns(pl.lit(f.stem).alias("vehicle_id"))
        dfs.append(df)

    trajectories = (
        pl.concat(dfs)
        .rename(
            {
                "unix time": "global_time",
                " postmile": "x_meters",
                " speed": "velocity"
            }
        )
        .select(
            pl.col.vehicle_id,
            pl.col.x_meters * 1609.34,  # miles to meters
            (pl.col.global_time * 1000).cast(pl.Datetime("ms")),
            pl.col.velocity / 2.23694, # convert from milesph to m/s
            )
        .pipe(add_relative_time).pipe(create_unique_vehicle_id)
        # .pipe(clip_trajectories, xmin=33_000, xmax=42_000) # cut after last exit
        # .pipe(clip_trajectories, xmin=33_000, xmax=46_000) # full highway
        # .pipe(clip_trajectories, xmin=35_500, xmax=38_000) # homogeneous section
        .pipe(clip_trajectories, xmin=xmin, xmax=xmax) # custom section
        # .select(["global_time", "x_meters", "vehicle_id", "velocity"])
    )
    return trajectories

def read_raw_mobile_century_trajectories(trajectories_dir: str) -> pl.DataFrame:
    csv_dir = Path(trajectories_dir)
    files = sorted(csv_dir.glob("*.csv"))

    dfs = []
    for f in files:
        df = pl.read_csv(f).with_columns(pl.lit(f.stem).alias("vehicle_id"))
        dfs.append(df)

    trajectories = (
        pl.concat(dfs)
        .rename(
                {
                    "unixtime": "record_timestamp",
                    "speed": "velocity"
                }
            )
        .with_columns(
            pl.col.vehicle_id,
            pl.col.record_timestamp.cast(pl.Datetime("ms")),
            pl.col.velocity / 2.23694, # convert from milesph to m/s
            )
    )
    return trajectories
