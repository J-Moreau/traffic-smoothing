import numpy as np
import polars as pl


def clip_trajectories(
    trajectories: pl.DataFrame,
    xmin: float,
    xmax: float,
    start_seconds: float=0,
    end_seconds: float=np.inf,
) -> pl.DataFrame:
    # make sure to use strict inequalities to avoid including the last point
    trajectories_clip_time = trajectories.filter(
        pl.col.time_seconds.is_between(start_seconds, end_seconds, closed="left")
    )
    return (
        trajectories_clip_time.filter(pl.col.x_meters > xmin).filter(
            pl.col.x_meters < xmax
        )
    ).with_columns(
        x_meters=pl.col.x_meters - xmin,
        time_seconds=pl.col.time_seconds - start_seconds,
    )


def add_relative_time(trajectories: pl.DataFrame) -> pl.DataFrame:
    """
    Add a column with the time in seconds since the start of the trajectory.
    """
    return trajectories.with_columns(
        time_seconds=(
            (
                pl.col.global_time - pl.col.global_time.min()
            ).dt.total_milliseconds()
            / 1000
        )
    ).with_columns(time_minutes=pl.col.time_seconds / 60)


def downsample_trajectories(
    trajectories: pl.DataFrame,
    fraction: float = 0.005,
    every_second: int = 3,
    seed: int = 0,
) -> pl.DataFrame:
    """
    sample a fraction of vehicles at given sample rate
    (every = 3 seconds)
    """
    per_vehicle = (
        trajectories.sort(["vehicle_id", "global_time"])
        .group_by("vehicle_id", maintain_order=True)
        .all()
        .sample(fraction=fraction, seed=seed)
    )
    np.random.seed(seed)
    random_offset = np.random.uniform(0, 3, size=per_vehicle.height)
    return (
        per_vehicle
        # add a random offset so that the samples are not synchronized
        .with_columns(random_offset=random_offset)
        .explode(pl.exclude("vehicle_id", "random_offset"))
        .with_columns(
            global_time=pl.col("global_time")
            + pl.duration(seconds=pl.col("random_offset"))
        )
        .group_by_dynamic(
            "global_time", every=f"{every_second}s", group_by="vehicle_id"
        )
        .agg(pl.all().first())
        .with_columns(
            global_time=pl.col("global_time")
            - pl.duration(seconds=pl.col("random_offset")),
        )
        .drop("random_offset")
        # .filter(pl.col.lane)
    )


def create_unique_vehicle_id(trajectories: pl.DataFrame, delta_x_meters=100):
    """
    Some Vehicle_id may be reused later and reenter the road at the start
    we instead count the number of returns to keep one id per trajectory

    delta_x_meters is the threshold difference for a return
    """
    return (
        trajectories.with_columns(
            is_return=(
                pl.col.x_meters.diff().over("vehicle_id", order_by="global_time")
                < -delta_x_meters
            ).fill_null(False)
        )
        .with_columns(
            num_returns=pl.col.is_return.cum_sum().over(
                "vehicle_id", order_by="global_time"
            )
        )
        # .with_columns(
        #     min_time=
        #     # Problem: We need to breaks ties in case two vehicles enter at the same time
        #     (pl.col.global_time.cast(pl.Int64) # ms
        #     .min().over(["vehicle_id", "num_returns"])
        # )
        # .with_columns(
        #     vehicle_id=pl.col.min_time.rank()
        # )
        .with_columns(
            vehicle_id=pl.col.vehicle_id + "_" + pl.col.num_returns.cast(pl.Utf8)
        )
        .drop("is_return", "num_returns")
    )