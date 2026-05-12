from dataclasses import dataclass, field
from typing import Literal, Optional

from traffic_models.var_optim import FourDVarConfig


@dataclass
class DatasetConfig:
    name: str = "us-101"
    path: str = "data/ngsim/NGSIM_trajectories.parquet"
    start_seconds: int = 180  # start 3 minutes after the beginning
    end_seconds: int = 2700
    lanes: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    probe_fraction: float = 0.02
    is_using_probe: bool = True
    boundary_fraction: float = 0.2
    smoothing: bool = False
    xmin_meters: Optional[float] = None
    xmax_meters: Optional[float] = None
    ramps: bool = True

@dataclass
class KalmanConfig:
    velocity_model_variance: float = 0.1
    desroziers: bool = False

@dataclass
class FourDVarQuickConfig:
    velocity_model_variance: float = 0.01
    init_variance: float = 1.0
    n_iters: int = 1000
    solver: str = "rusanov"
    n_windows: int = 1
    init: str = "asm" # or "forward_fill" or "naive_rollout"
    learn_flow: bool = True
    fundamental_diagram: str = "ARZPiecewiseQuadratic" # can't use Literal with OmegaConf

@dataclass
class ProbeExperimentConfig:
    name: str = "default"
    dt_seconds: int = 1
    dx_meters: int = 50
    seed: int = 0
    data: DatasetConfig = field(default_factory=DatasetConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    fourdvar: FourDVarQuickConfig = field(default_factory=FourDVarQuickConfig)