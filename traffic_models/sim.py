"""General Physics simulation utils"""
from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass
class DiscretizationGrid:
    dx_meters: float
    dt_seconds: float
    n_cells: int
    n_timesteps: int

    @staticmethod
    def from_dimensions(
        dx_meters: float, dt_seconds: float, lmax:float, tmax:float
    ) -> "DiscretizationGrid":
        n_cells = int(ceil(lmax / dx_meters))
        n_timesteps = int(ceil(tmax / dt_seconds))
        return DiscretizationGrid(
            dx_meters=dx_meters,
            dt_seconds=dt_seconds,
            n_cells=n_cells,
            n_timesteps=n_timesteps,
        )


@dataclass 
class TimeGrid:
    """Time discretization grid for CTM simulations"""
    dt_seconds: float
    n_timesteps: int

    @staticmethod
    def from_duration(dt_seconds: float, tmax: float) -> "TimeGrid":
        n_timesteps = int(ceil(tmax / dt_seconds))
        return TimeGrid(
            dt_seconds=dt_seconds,
            n_timesteps=n_timesteps,
        )

@dataclass
class RampConfig:
    on_ramps_index: np.ndarray
    off_ramps_index: np.ndarray
    # merge_ratio: np.ndarray
    # diverge_ratio: np.ndarray