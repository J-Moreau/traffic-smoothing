import polars as pl


def discretize_trajectories(
    sample_trajectories: pl.DataFrame,
    DT_SECONDS: float,
    DX_METERS: float,
    N_CELLS: int,
    N_TIMESTEPS: int,
) -> pl.DataFrame:
    """
    Discretize time and space into bins
    Use mean x and velocity values OVER VEHICLE ID for each temporal bin
    """
    return (
        sample_trajectories.sort("time_seconds")
        .with_columns(
            t_index=(pl.col.time_seconds/ DT_SECONDS).floor().cast(pl.Int64),
        )
        .group_by("t_index","vehicle_id")
        .agg(pl.col.time_seconds.last(),pl.col.x_meters.last(), pl.col.velocity.mean())
        .with_columns(
            x_index=(pl.col.x_meters/ DX_METERS).floor().cast(pl.Int64),
        )
        .filter(
            pl.col.x_index < N_CELLS,
            pl.col.t_index < N_TIMESTEPS,
        )
    )