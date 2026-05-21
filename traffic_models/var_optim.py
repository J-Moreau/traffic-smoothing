"""Variational Data Assimilation"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from numpy.typing import NDArray
from scipy.signal import convolve2d

import traffic_models.flows as numpy_implem
import traffic_models.nn.godunov_torch as gdnv_torch
from traffic_models.dense_fields import forward_fill_nan_columns, interpolate_nan_matrix
from traffic_models.extended_kalman import run_ekf_on_trajectories, run_rts_smoother
from traffic_models.flows_default import (
    greenshields_mobile_century_herrera_bayen_2010,
    triangular_mobile_century_herrera_bayen_2010,
)
from traffic_models.naive_physics import naive_physics_on_trajectories
from traffic_models.nn.arz_torch import (
    ARZFlow,
    arz_lax_friedrichs_ctm_step,
    arz_solver_batch,
    arz_solver_rollout,
)
from traffic_models.nn.asm import asm_initial_guess
from traffic_models.nn.flows_torch import (
    ConvexVelocityFlow,
    GreenshieldsFlow,
    TriangularFlow,
    torch_implem_from_numpy,
)
from traffic_models.nn.lax_friedrichs_torch import lax_friedrichs_step
from traffic_models.sim import DiscretizationGrid, RampConfig


@dataclass
class FourDVarConfig:
    background_variance: float
    model_variance: float
    measurement_variance: float
    n_iters: int 
    solver: Literal["godunov", "lax_friedrichs", "rusanov"] = "rusanov"


@dataclass
class FourDVarResult:
    v_smooth: torch.Tensor
    history: pl.DataFrame
    flow: Optional[nn.Module] = None
    rho: Optional[torch.Tensor] = None
    flux_ratio: Optional[torch.Tensor] = None
    source_term: Optional[torch.Tensor] = None


@dataclass
class WindowedFourDVarResult:
    velocity_hat: np.ndarray
    velocity_pred: np.ndarray
    history: list[pl.DataFrame]
    flow: Optional[nn.Module] = None
    rho_hat: Optional[np.ndarray] = None
    rho_pred: Optional[np.ndarray] = None


def trajectories_to_matrix(trajectories: pl.DataFrame, n_timesteps: int, n_cells: int):
    grouped_trajectories = trajectories.group_by(["t_index", "x_index"]).agg(
        pl.col.velocity.mean()
    )
    t_indexes = grouped_trajectories["t_index"]
    node_indexes = grouped_trajectories["x_index"]
    velocities = grouped_trajectories["velocity"].to_numpy()
    velocities_masked = np.full((n_timesteps, n_cells), np.nan)
    velocities_masked[t_indexes, node_indexes] = velocities
    return velocities_masked


def windowed_w4DVAR(
    trajectories: pl.DataFrame,
    v_0: NDArray,
    conf: FourDVarConfig,
    grid: DiscretizationGrid,
    flow: numpy_implem.GreenshieldsFlow | numpy_implem.TriangularFlow | numpy_implem.ARZFlow,
    window_seconds: float,
    ramp_config: RampConfig,
    learn_flow: bool = False,
    device="cpu",
    init: Literal["forward_fill", "asm", "naive_rollout", "rts", "naive_interp", "naive_smoothing"] = "asm",
    forecast: bool = False,
):
    """
    applies weak constraint 4DVar iteratively

    v_0 is the initial background state (n_cells,)
    trajectories contains discretized measurements (columns: t_index, x_index, velocity)
    """
    v_background = torch.tensor(v_0, dtype=torch.float, device=device)
    v_start = v_background
    n_windows = int(np.ceil((grid.n_timesteps * grid.dt_seconds) / window_seconds))
    window_length = int(np.ceil(window_seconds / grid.dt_seconds))
    # init outputs
    history = []
    velocity_hat = np.empty((grid.n_timesteps, grid.n_cells), dtype=np.float32)
    rho_hat = np.empty((grid.n_timesteps, grid.n_cells), dtype=np.float32)
    velocity_pred = np.empty((grid.n_timesteps, grid.n_cells), dtype=np.float32)
    ramp_idx = torch.tensor(
        np.concatenate([ramp_config.on_ramps_index, ramp_config.off_ramps_index]),
        dtype=torch.int64,
        device=device,
    )
    # ramp_idx = torch.cat((_ramp_idx, _ramp_idx + 1)).clamp(max=v_0.shape[0]-2)  # add the next cell 
    torch_flow = torch_implem_from_numpy(flow, frozen=not learn_flow).to(device)
    flux_ratio = torch.ones(ramp_idx.shape[0], device=device, requires_grad=False)
    source_term = torch.zeros(ramp_idx.shape, device=device, requires_grad=False)
    if isinstance(torch_flow, ARZFlow):
        rho_background = torch_flow.rho_eq(v_background)
        rho_start = rho_background
    assert n_windows > 0
    for i in range(n_windows):
        warmup_length = min(i*window_length, window_length//10)  # warmup time of the window to mitigate border effects
        idx_start = i*window_length
        idx_background = idx_start - warmup_length
        idx_end = min(grid.n_timesteps, (i + 1) * window_length)
        window_grid = DiscretizationGrid(
            dt_seconds=grid.dt_seconds,
            dx_meters=grid.dx_meters,
            n_cells=grid.n_cells,
            n_timesteps=idx_end - idx_background,
        )
        # reset flow parameters
        torch_flow = torch_implem_from_numpy(flow, frozen=not learn_flow).to(device)
        with torch.no_grad():
            if forecast:
                if isinstance(torch_flow, ARZFlow):
                    q_init = torch_flow.q(rho_start, v_start)
                    U_init = torch.stack([rho_start, q_init], dim=0)
                    U_pred = arz_solver_rollout(
                        U_init,
                        torch_flow,
                        window_grid,
                        ramp_idx,
                        flux_ratio,
                        source_term,
                        solver=conf.solver,
                        # solver="godunov"
                    )
                    v_pred = torch_flow.v(U_pred[:, 0, :], U_pred[:, 1, :])
                else:
                    rho_init = torch_flow.density_from_velocity(v_start)
                    rho_pred = lwr_solver_rollout(
                        rho_init, torch_flow, window_grid, solver=conf.solver
                    )
                    v_pred = torch_flow.velocity_from_density(rho_pred)
            # skip the forecast if not needed to save computation time
            else:
                v_pred = torch.zeros((window_grid.n_timesteps, window_grid.n_cells), device=device)

        traj_in_window = trajectories.filter(
            pl.col.t_index.is_between(idx_background, idx_end, closed="left")
        ).with_columns(pl.col.t_index - idx_background)
        t_index = traj_in_window["t_index"].to_numpy()
        x_index = traj_in_window["x_index"].to_numpy()
        v_meas = traj_in_window["velocity"].to_numpy()
        masked_matrix = trajectories_to_matrix(
            traj_in_window, window_grid.n_timesteps, grid.n_cells
        )

        # Option 1: simple forward fill
        if init=="forward_fill":
            v_init = forward_fill_nan_columns(masked_matrix, start_fill_value=v_background.detach().cpu().numpy())

        # Option 1.5: same but interpolates forward and backward
        elif init=="naive_interp":
            v_init = interpolate_nan_matrix(masked_matrix)

        # Option 2: adaptive smoothing
        elif init=="asm":
            v_init = asm_initial_guess(
                masked_matrix,
                dx_meters=grid.dx_meters,
                dt_seconds=window_grid.dt_seconds,
            )
        # Option 2.5: non adaptive isotropic smoothing
        elif init=="naive_smoothing":
            v_init = asm_initial_guess(
                masked_matrix,
                dx_meters=grid.dx_meters,
                dt_seconds=window_grid.dt_seconds,
                v_cong=10_000.0, # large propagation speed decouples space and time, making the kernel isotropic
                v_free=10_000.0,
            )
        # Option 3: naive rollout
        elif init=="naive_rollout":
            _flow = greenshields_mobile_century_herrera_bayen_2010() # this works ok for naive rollout
            v_init = naive_physics_on_trajectories(traj_in_window, v_background.detach().cpu().numpy(), _flow, window_grid, trust_ratio=1.0).v_hat
        
        elif init=="rts":
            _flow = triangular_mobile_century_herrera_bayen_2010() # this one is better for RTS
            _pred = run_ekf_on_trajectories(
                discretized_trajectories=traj_in_window,
                P_0=conf.background_variance * np.eye(grid.n_cells),
                v_0=v_background.detach().cpu().numpy(),
                flow=_flow,
                Q=conf.model_variance * np.eye(grid.n_cells),
                VELOCITY_MEASURE_VARIANCE=conf.measurement_variance,
                grid=window_grid,
            )
            v_init, _ = run_rts_smoother(_pred, Q=conf.model_variance * np.eye(grid.n_cells), flow=_flow)

        result = weak_constraint_4DVAR(
            torch.tensor(v_init, dtype=torch.float, device=device),
            v_background,
            torch.tensor(t_index, dtype=torch.int, device=device),
            torch.tensor(x_index, dtype=torch.int, device=device),
            torch.tensor(v_meas, dtype=torch.float, device=device),
            torch_flow,
            conf,
            window_grid,
            flux_ratio=None, # reset flux ratio at each window, otherwise errors can accumulate and optimization diverges
            source_term=None,
            learn_flow=learn_flow,
            ramp_idx=ramp_idx,
        )
        v_smooth = result.v_smooth
        flux_ratio = result.flux_ratio.detach()
        source_term = result.source_term.detach() #if result.source_term is not None else source_term
        history.append(result)
        velocity_hat[idx_start:idx_end] = v_smooth[warmup_length:].detach().cpu().numpy()
        velocity_pred[idx_start:idx_end] = v_pred[warmup_length:].clone().cpu().numpy()
        next_warmup_length = window_length//10
        v_background = v_smooth[-next_warmup_length-1, :].detach()
        v_start = v_smooth[-1, :].detach()
        if isinstance(torch_flow, ARZFlow):
            rho_hat[idx_start:idx_end] = result.rho[warmup_length:].clone().cpu().numpy()
            rho_background = result.rho[-next_warmup_length-1, :].detach()
            rho_start = result.rho[-1, :].detach()
    return WindowedFourDVarResult(
        velocity_hat, velocity_pred, history, torch_flow, rho_hat=rho_hat
    )


def weak_constraint_4DVAR(
    v_init: torch.Tensor,
    v_background: torch.Tensor,
    t_index: torch.Tensor,
    x_index: torch.Tensor,
    v_meas: torch.Tensor,
    flow: GreenshieldsFlow | TriangularFlow | ARZFlow,
    conf: FourDVarConfig,
    grid: DiscretizationGrid,
    learn_flow: bool = False,
    ramp_idx: Optional[torch.Tensor] = None,
    flux_ratio: Optional[torch.Tensor] = None,
    source_term: Optional[torch.Tensor] = None,
):
    """
    Minimizes

    J(v) = ||v[0] - v_background||^2_B + ||v[t+1] - F(v[t])||^2_Q + ||v - v_meas||^2_R
    where   B is background cov
            Q is model cov
            R is measure cov

    Arguments:
        v_init: first guess (n_timesteps, n_cells)
        v_background: background state at t=0 (n_cells,)
        t_index: time indices of measurements (n_meas,)
        x_index: space indices of measurements (n_meas,)
        v_meas: measurements (n_meas,)

    """
    measurement_ratio = v_meas.shape[0] / (grid.n_cells * grid.n_timesteps)
    background_ratio = 1 / grid.n_timesteps

    model_variance = conf.model_variance
    background_variance = conf.background_variance*background_ratio**2
    measurement_variance = conf.measurement_variance*measurement_ratio**2

    device = v_init.device
    v = v_init.clone()
    v.requires_grad_()
    if ramp_idx is None:
        ramp_idx = torch.empty((0,), dtype=torch.int64, device=device)
    if flux_ratio is None:
        flux_ratio = (torch.ones(ramp_idx.shape[0], device=device))
    flux_ratio.requires_grad_(True)
    if source_term is None:
        source_term = torch.zeros(ramp_idx.shape, device=device)
    source_term.requires_grad_(True)
    if isinstance(flow, ARZFlow):
        rho_ = flow.rho_eq(v.detach()).detach().requires_grad_()

    # optimizer = torch.optim.LBFGS(params=[v], lr=1.0)
    optimizer = torch.optim.Adam(
        [
            {"params": [v], "lr": 0.1 * flow.v_max.detach().cpu().item()},
            {"params": [flux_ratio], "lr": 0.01},
            {"params": [source_term], "lr": 0.01/grid.dx_meters*(flow.rho_max*flow.v_max/4).detach().cpu().item()},
        ]
    )
    if learn_flow:
        optimizer.add_param_group({"params": flow.parameters(), "lr": 0.01})
    if isinstance(flow, ARZFlow):
        optimizer.add_param_group(
            {"params": [rho_], "lr": 0.1 * flow.rho_max.detach().cpu().item()}
        )
    def closure():
        optimizer.zero_grad()

        if isinstance(flow, ConvexVelocityFlow):
            v_pred = gdnv_torch.convex_godunov_step(
                v, flow, flow.v_c, grid.dt_seconds, grid.dx_meters
            )
        elif isinstance(flow, ARZFlow):
            q = flow.q(rho_, v)
            U = torch.stack([rho_, q], dim=1)
            # U_pred = arz_solver_batch(U, flow, grid, solver=conf.solver)
            U_pred = arz_solver_batch(
                U,
                flow,
                grid,
                ramp_idx=ramp_idx,
                flux_ratio=flux_ratio,
                solver=conf.solver,
            )
            U_pred[U_pred.isnan()] = U[U_pred.isnan()]
            U_pred[U_pred.isinf()] = U[U_pred.isinf()]
            U_pred[:, 1, :].clamp_(min=0)
            rho_pred = U_pred[:, 0, :]
            q_pred = U_pred[:, 1, :]
            rho_pred[:, ramp_idx] += source_term * grid.dt_seconds
            # q_pred <- q + F * w where w = h + v
            # q_pred[:, ramp_idx] += source_term * (q_pred[:, ramp_idx]/rho_pred[:, ramp_idx].clamp(1e-6)) * grid.dt_seconds
            v_pred = flow.v(
                rho_pred.clamp(min=1e-6, max=flow.rho_max.detach().cpu().item() * 0.999),
                q_pred,
            )
        else:
            rho = flow.density_from_velocity(v)
            rho_pred = lwr_solver_batch(rho, flow, grid, solver=conf.solver)
            rho_pred[:, ramp_idx] += source_term * grid.dt_seconds
            v_pred = flow.velocity_from_density(rho_pred)

        background_cost = (v[0] - v_background).pow(2).sum() / background_variance
        model_cost = (v[1:] - v_pred[:-1]).pow(2).sum() / model_variance
        if isinstance(flow, ARZFlow):
            model_cost += (
                (rho_[1:] - rho_pred[:-1]).pow(2).sum()
                / model_variance
                / flow.rho_max**2
                * flow.v_max**2
            )
        # model_cost += 5 * (rho_[1:] - rho_[:-1]).pow(2).sum() / conf.model_variance / flow.rho_max**2 * flow.v_max**2 # increase diffusion
        model_cost += ((v[1:] - v[:-1])*(v[:-1]>flow.v_max/2)).pow(2).sum() / conf.model_variance  # increase diffusion

        measurement_cost = (v[t_index, x_index] - v_meas).pow(
            2
        ).sum() / measurement_variance
        loss = background_cost + model_cost + measurement_cost
        loss.backward()
        return background_cost, model_cost, measurement_cost

    history = []
    for i in range(conf.n_iters):
        background_cost, model_cost, measurement_cost = optimizer.step(closure)
        loss = background_cost + model_cost + measurement_cost
        with torch.no_grad():
            if isinstance(flow, ARZFlow):
                rho_.clamp_(min=0.0001, max=flow.rho_max.detach().cpu().item() * 0.999)
            v.clamp_(min=1e-6, max=flow.v_max.detach().cpu().item() * 0.999)
            # flux_ratio.clamp_(min=0.8, max=1.2)
            flux_ratio.clamp_(min=0.8, max=1.2)
        history.append(
            dict(
                step=i,
                loss=loss.detach().item(),
                **{
                    name: param.detach().clone().cpu().numpy()
                    if param.ndim == 0
                    else param.detach().clone().cpu()
                    for name, param in flow.named_parameters()
                },
                flux_ratio=flux_ratio.cpu().detach().clone().numpy().tolist(),
                source_term=source_term.cpu().detach().clone().numpy().tolist(),
                background_cost=background_cost.detach().item(),
                model_cost=model_cost.detach().item(),
                measurement_cost=measurement_cost.detach().item(),
            )
        )
    return FourDVarResult(
        v.detach(),
        pl.DataFrame(history),
        flow,
        rho=rho_.detach().clone() if isinstance(flow, ARZFlow) else None,
        flux_ratio=flux_ratio.detach().clone(),
        source_term=source_term.detach().clone(),
    )


def lwr_solver_rollout(
    rho_init: torch.Tensor,
    flow: GreenshieldsFlow | TriangularFlow,
    grid: DiscretizationGrid,
    solver: Literal["godunov", "lax_friedrichs"],
) -> torch.Tensor:
    """
    Solves the PDE iteratively
    rho_t + flow(rho)_x = 0
    """
    rho_t = rho_init.clone()
    output = [rho_t]
    for i in range(grid.n_timesteps - 1):
        # prediction step
        if solver == "godunov":
            output.append(
                gdnv_torch.godunov_step(
                    output[i],
                    Q=flow,
                    rho_c=flow.rho_c,
                    dt=grid.dt_seconds,
                    dx=grid.dx_meters,
                ).clone()
            )

        elif solver == "lax_friedrichs":
            # should be the same for convex or concave flow
            output.append(
                lax_friedrichs_step(
                    flow, output[i], grid.dt_seconds, grid.dx_meters
                ).clone()
            )
        else:
            raise ValueError(
                f"Unknown solver {solver}, must be 'godunov' or 'lax_friedrichs'"
            )
    return torch.stack(output, dim=0)


def lwr_solver_batch(
    rho_previous: torch.Tensor,
    flow: GreenshieldsFlow | TriangularFlow,
    grid: DiscretizationGrid,
    solver: Literal["godunov", "lax_friedrichs"],
):
    """Applies one step of solving"""
    # prediction step
    if solver == "godunov":
        return gdnv_torch.godunov_step(
            rho_previous,
            Q=flow,
            rho_c=flow.rho_c,
            dt=grid.dt_seconds,
            dx=grid.dx_meters,
        )
    elif solver == "lax_friedrichs":
        return lax_friedrichs_step(flow, rho_previous, grid.dt_seconds, grid.dx_meters)
