from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import scipy.interpolate
import scipy.ndimage
import torch

from traffic_models.nn.asm import AdaptiveSmoothing


@dataclass
class FlowData:
    speed: np.ndarray
    density: np.ndarray
    flow: np.ndarray
    xmax: float
    tmax: float
    dt_seconds: float
    dx_meters: float
    sparse: bool = False

def remove_missing_values(trajectories: pl.DataFrame) -> pl.DataFrame:
    """
    Remove rows with missing values in space_headway or time_headway.
    """
    try:
        return trajectories.filter(
            (pl.col.space_headway > 0) & (pl.col.time_headway <= 9999)
        )
    except pl.ColumnNotFoundError:
        return trajectories

def mean_values_over_grid(
    trajectories_with_index: pl.DataFrame,
    col_names: list[str] = ["velocity", "space_headway", "time_headway"],
) -> pl.DataFrame:
    return (
        trajectories_with_index
        # consider one data point per vehicle not to overrepresent slower vehicles
        .group_by("time_index", "space_index", "vehicle_id")
        .agg(*[pl.col(name).mean().alias(name) for name in col_names])
        .group_by("time_index", "space_index")
        .agg(*[pl.col(name).mean().alias(f"mean_{name}") for name in col_names])
    )


def aggregate_fields_over_grid(trajectories_with_index: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate signals over time and space index to get
    - mean velocity
    - number of signals
    - spatial proportion of signals
    - number of vehicles crossing

    """
    return (
        trajectories_with_index.sort("vehicle_id", "time_seconds", "x_meters")
        .with_columns(
            last_index=pl.col.space_index.shift(1).over("vehicle_id").fill_null(-1),
            n_samples_per_vehicle=pl.len().over(
                "time_index", "space_index", "vehicle_id"
            ),
        )
        .with_columns(
            arrivals=pl.when((pl.col.space_index > pl.col.last_index))
            .then(1)
            .otherwise(0)
        )
        .group_by("time_index", "space_index")
        .agg(
            n_vehicles_crossing=pl.col("arrivals").sum(),
            n_signals=pl.count(),
            # consider one data point per vehicle not to overrepresent slower vehicles
            # this is a weighted mean with weight 1/nb of data points per vehicle
            mean_velocity=(pl.col.velocity / pl.col.n_samples_per_vehicle).sum()
            / (1 / pl.col.n_samples_per_vehicle).sum(),
            mean_spacing=(pl.col.space_headway / pl.col.n_samples_per_vehicle).sum()
            / (1 / pl.col.n_samples_per_vehicle).sum(),
            mean_interval=(pl.col.time_headway / pl.col.n_samples_per_vehicle).sum()
            / (1 / pl.col.n_samples_per_vehicle).sum(),
        )
        # add the proportion of signals in the space bin relative to the total number of signals
        .with_columns(
            signal_proportion=pl.col.n_signals
            / pl.col.n_signals.sum().over("time_index"),
        )
    )


def smooth_space_and_time(
    grid_elements: pl.DataFrame,
    TIME_INDEX_SMOOTHING: int,
    SPACE_INDEX_SMOOTHING: int,
    col_names: list[str] = ["mean_velocity", "mean_space_headway", "mean_time_headway"],
) -> pl.DataFrame:
    return (
        grid_elements.sort("time_index", "space_index")
        .rolling(
            "time_index", period=f"{TIME_INDEX_SMOOTHING}i", group_by=pl.col.space_index
        )
        .agg(*[pl.col(name).mean().alias(name) for name in col_names])
        .rolling(
            "space_index",
            period=f"{SPACE_INDEX_SMOOTHING}i",
            group_by=pl.col.time_index,
        )
        .agg(*[pl.col(name).mean().alias(name) for name in col_names])
    )


def interpolate_nan_matrix(matrix):
    x, y = np.indices(matrix.shape)
    valid_mask = ~np.isnan(matrix)
    interpolated = scipy.interpolate.griddata(
        (x[valid_mask], y[valid_mask]),
        matrix[valid_mask],
        (x, y),
        method='nearest'
    )
    return interpolated

def make_anisotropic_kernel(sigma: float, tau: float, v: float, dx:float, dt:float, kernel_deviations:float=3, mode:Literal["gaussian","exponential"]="gaussian") -> np.ndarray:
    kd=kernel_deviations
    x = np.linspace(-kd*sigma, kd*sigma, int(2*kd*sigma/dx))
    t = np.linspace(-kd*tau, kd*tau, int(2*kd*tau/dt))
    xx, tt = np.meshgrid(x, t)
    if mode == "exponential":
        kernel = np.exp(-0.5 * (np.abs(xx) / sigma + np.abs(tt - xx / v) / tau))
    else:
        kernel = np.exp(-0.5 * (xx**2 / sigma**2 + (tt - xx / v)**2 / tau**2))
    kernel = np.maximum(kernel, 1e-9)
    return kernel / kernel.sum()

def adaptive_smoothing_method(
    matrix: np.ndarray, dx:float, dt:float, sigma_x=50.0, sigma_t=60.0,
    kernel_deviations:float=3,
    v_threshold: float = 15.0, v_free: float = 20.0, v_cong: float = -8.0,
    delta_v: float = 6.0,
) -> np.ndarray:
    sigma_free, tau_free = sigma_x, sigma_t
    sigma_cong, tau_cong = sigma_x, sigma_t
    
    valid_mask = ~np.isnan(matrix)
    matrix_filled = np.where(valid_mask, matrix, 0.0)
    
    
    kernel_free = make_anisotropic_kernel(sigma_free, tau_free, v_free, dx, dt, kernel_deviations)
    kernel_cong = make_anisotropic_kernel(sigma_cong, tau_cong, v_cong, dx, dt, kernel_deviations)
    
    weights_free = scipy.ndimage.correlate(valid_mask.astype(float), kernel_free, mode='nearest')
    weights_cong = scipy.ndimage.correlate(valid_mask.astype(float), kernel_cong, mode='nearest')
    
    smoothed_free = scipy.ndimage.correlate(matrix_filled, kernel_free, mode='nearest') / np.maximum(weights_free, 1e-10)
    smoothed_cong = scipy.ndimage.correlate(matrix_filled, kernel_cong, mode='nearest') / np.maximum(weights_cong, 1e-10)
    
    # low_speed_mask = np.where(valid_mask, (matrix < v_threshold).astype(float), 0.0)
    # weight_cong = scipy.ndimage.convolve(low_speed_mask, kernel_cong, mode='nearest') / np.maximum(weights_cong, 1e-10)
    
    weight_cong = 0.5 * (1 + np.tanh((np.minimum(smoothed_cong, smoothed_free) - v_threshold) / delta_v)) # sharpen the transition between free and congested weights
    result = weight_cong * smoothed_cong + (1 - weight_cong) * smoothed_free
    return np.where((weights_free>1e-10) & (weights_cong>1e-10), result, np.nan)

def adaptive_smoothing_method_fast(
    matrix: np.ndarray,
    dx: float,
    dt: float,
    scale_t: int = 10,
    scale_x: int = 2,
    *args,
    **kwargs
) -> np.ndarray:
    import warnings
    scale_t = min(matrix.shape[0], scale_t)
    scale_x = min(matrix.shape[1], scale_x)
    t_bins = matrix.shape[0] // scale_t
    x_bins = matrix.shape[1] // scale_x
    coarse_dt = dt * scale_t
    coarse_dx = dx * scale_x
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        matrix_coarse = np.nanmean(
            matrix[:t_bins * scale_t, :x_bins * scale_x]
            .reshape(t_bins, scale_t, x_bins, scale_x),
            axis=(1, 3)
        )
    
    smoothed_coarse = adaptive_smoothing_method(
        matrix_coarse, dx=coarse_dx, dt=coarse_dt, *args, **kwargs
    )
    
    zoom_factors = (matrix.shape[0] / smoothed_coarse.shape[0], matrix.shape[1] / smoothed_coarse.shape[1])
    return scipy.ndimage.zoom(smoothed_coarse, zoom_factors, order=1)

# Source - https://stackoverflow.com/a
# Posted by Divakar
# Retrieved 2025-12-17, License - CC BY-SA 4.0

def forward_fill_nan_columns(a:np.ndarray, start_fill_value:float|np.ndarray=0):
    mask = np.isnan(a)
    tmp = a[0].copy()
    if isinstance(start_fill_value, (int, float)):
        start_fill_value = np.full(a.shape[1], start_fill_value)
    a[0][mask[0]] = start_fill_value[mask[0]]
    mask[0] = False
    idx = np.where(~mask,np.arange(mask.shape[0])[:,None],0)
    out = np.take_along_axis(a,np.maximum.accumulate(idx,axis=0),axis=0)
    a[0] = tmp
    return out

def agg_trajectories_into_field(
    trajectories: pl.DataFrame,
    xmax: float,
    tmax: float,
    space_bin_meters: int,
    time_bin_seconds: float,
    smoothing: bool,
) -> FlowData:
    # Define grid parameters
    SPACE_BIN_METERS = space_bin_meters
    TIME_BIN_SECONDS = time_bin_seconds
    LEN_X_GRID = int(np.ceil((xmax) / SPACE_BIN_METERS))
    LEN_T_GRID = int(np.ceil((tmax) / TIME_BIN_SECONDS))

    # discretize time and space into bins
    trajectories_with_index = trajectories.with_columns(
        time_index=(pl.col.time_seconds / TIME_BIN_SECONDS).floor().cast(pl.Int32),
        space_index=(pl.col.x_meters / SPACE_BIN_METERS).floor().cast(pl.Int32),
    )

    # TIME_SMOOTHING_SECONDS = 5  # rolling mean window
    # TIME_INDEX_SMOOTHING = int(TIME_SMOOTHING_SECONDS / TIME_BIN_SECONDS)
    # SPACE_SMOOTHING_METERS = 50  # rolling mean window
    # SPACE_INDEX_SMOOTHING = int(SPACE_SMOOTHING_METERS / SPACE_BIN_METERS)

    TIME_INDEX_SMOOTHING = 5
    SPACE_INDEX_SMOOTHING = 1

    col_names = set(["velocity", "space_headway", "time_headway"]).intersection(trajectories_with_index.columns)
    mean_col_names = [f"mean_{name}" for name in col_names]
    agg_grid = trajectories_with_index.pipe(remove_missing_values).pipe(
        mean_values_over_grid, col_names=list(col_names)
    )
    # if smoothing:
    #     agg_grid = agg_grid.pipe(
    #         smooth_space_and_time, TIME_INDEX_SMOOTHING, SPACE_INDEX_SMOOTHING, mean_col_names
    #     )
    # Create grid arrays
    grid_speed = np.full((LEN_T_GRID, LEN_X_GRID), np.nan)
    grid_flow = np.full((LEN_T_GRID, LEN_X_GRID), np.nan)
    grid_density = np.full((LEN_T_GRID, LEN_X_GRID), np.nan)
    if "mean_time_headway" in agg_grid.columns:
        grid_flow[agg_grid["time_index"], agg_grid["space_index"]] = (
            1 / agg_grid["mean_time_headway"]
        )
    if "mean_space_headway" in agg_grid.columns:
        grid_density[agg_grid["time_index"], agg_grid["space_index"]] = (
            1 / agg_grid["mean_space_headway"]
        )
    grid_speed[agg_grid["time_index"], agg_grid["space_index"]] = agg_grid[
        "mean_velocity"
    ]
    if smoothing:
        kernel_time_window = 100*60
        kernel_space_window = 100*200
        # parameters for mobile century
        asm_model = AdaptiveSmoothing(
            kernel_time_window=kernel_time_window,
            kernel_space_window=kernel_space_window,
            dx=SPACE_BIN_METERS,
            dt=TIME_BIN_SECONDS,
            init_delta=10.0,
            init_tau=30.0,
        )
        measured_v_torch = torch.from_numpy(grid_speed).float().unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        asm_pred_torch = asm_model(measured_v_torch)
        grid_speed = asm_pred_torch.detach().numpy().squeeze(0)
    grid_speed = interpolate_nan_matrix(grid_speed)
    # grid_speed = forward_fill_nan_columns(grid_speed, start_fill_value=0)

    # Lane agnostic aggregation (not what is used in other papers)
    # grid_flow[agg_grid["time_index"], agg_grid["space_index"]] = (
    #     agg_grid["n_vehicles_crossing"] / TIME_BIN_SECONDS / N_LANES
    # )
    # grid_density[agg_grid["time_index"], agg_grid["space_index"]] = (
    #     agg_grid["n_signals"]  # nb of signals in time bin
    #     / N_LANES
    #     * SAMPLE_INTERVAL_MILLISECONDS
    #     / 1000
    #     / TIME_BIN_SECONDS
    #     / SPACE_BIN_METERS  # vehicles/meter/lane
    # )
    # Uncomment to use a sample-rate-agnostic aggregation instead
    # agg_grid = clipped_trajectories.with_columns(
    #         time_index=(pl.col.time_seconds / TIME_BIN_SECONDS).floor().cast(pl.Int32),
    #         space_index=(pl.col.x_meters / SPACE_BIN_METERS).floor().cast(pl.Int32),
    #     ).group_by("time_index", "space_index").agg(mean_velocity=pl.col("velocity").mean())
    # grid_speed[agg_grid["time_index"], agg_grid["space_index"]] = agg_grid[
    #     "mean_velocity"
    # ]

    return FlowData(
        speed=grid_speed,
        density=grid_density,
        flow=grid_flow,
        xmax=xmax,
        tmax=tmax,
        dt_seconds=TIME_BIN_SECONDS,
        dx_meters=SPACE_BIN_METERS,
    )

