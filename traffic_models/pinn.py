from dataclasses import dataclass

import matplotlib.pyplot as plt
import polars as pl
import torch
import torch.nn as nn
from numpy.typing import NDArray

import traffic_models.flows as numpy_implem
import wandb
from traffic_models.nn.arz_torch import ARZFlowBase
from traffic_models.nn.flows_torch import torch_implem_from_numpy
from traffic_models.sim import DiscretizationGrid


@dataclass
class PINNConfig:
    n_collocation: int = 10_000
    epochs: int = 10_000
    n_epochs_adam: int = 5000
    lr: float = 1e-3
    observation_weight: float = 10.0
    physics_weight: float = 1.0
    wandb_project: str = "pinn-traffic"
    log_every: int = 100
    learn_flow: bool = False


@dataclass
class PINNResult:
    network: nn.Module
    velocity_hat: NDArray
    rho_hat: NDArray
    flow: nn.Module


def _make_network(hidden: int = 60, n_layers: int = 8) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, 2))
    return nn.Sequential(*layers)


def _arz_residual(
    net: nn.Module,
    xt: torch.Tensor,
    flow: ARZFlowBase,
    x_max: float,
    t_max: float,
    diffusion_eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    xt.requires_grad_(True)
    out = net(xt)
    rho = torch.nn.functional.softplus(out[:, 0])
    v = torch.nn.functional.softplus(out[:, 1])

    grad_rho = torch.autograd.grad(rho.sum(), xt, create_graph=True)[0]
    drho_dt = grad_rho[:, 1] / t_max
    drho_dx = grad_rho[:, 0] / x_max

    grad_v = torch.autograd.grad(v.sum(), xt, create_graph=True)[0]
    dv_dt = grad_v[:, 1] / t_max
    dv_dx = grad_v[:, 0] / x_max

    # ARZ system in primitive variables:
    # ∂ρ/∂t + ∂(ρv)/∂x = eps ∂²ρ/∂x² (small diffusion for stability)
    # ∂v/∂t + (v - ρh'(ρ))∂v/∂x = (V_eq(ρ) - v)/τ
    continuity = drho_dt + rho * dv_dx + v * drho_dx
    if diffusion_eps > 0:
        grad2_rho = torch.autograd.grad(grad_rho[:, 0].sum(), xt, create_graph=True)[0]
        d2rho_dx2 = grad2_rho[:, 0] / x_max ** 2
        continuity = continuity - diffusion_eps * d2rho_dx2
    rho_dh = flow.rho_dh(rho)
    v_eq = flow.v_eq(rho)
    momentum = dv_dt + (v - rho_dh) * dv_dx - (v_eq - v) / flow.tau

    return continuity, momentum


def _lwr_residual(
    net: nn.Module,
    xt: torch.Tensor,
    flow: nn.Module,
    x_max: float,
    t_max: float,
    diffusion_eps: float = 1e-3,
) -> torch.Tensor:
    xt.requires_grad_(True)
    out = net(xt)
    rho = torch.nn.functional.softplus(out[:, 0])

    grad_rho = torch.autograd.grad(rho.sum(), xt, create_graph=True)[0]
    drho_dt = grad_rho[:, 1] / t_max

    Q = flow(rho)
    grad_Q = torch.autograd.grad(Q.sum(), xt, create_graph=True)[0]
    dQ_dx = grad_Q[:, 0] / x_max

    # LWR: ∂ρ/∂t + ∂Q(ρ)/∂x = eps ∂²ρ/∂x² (small diffusion for stability)
    continuity = drho_dt + dQ_dx
    if diffusion_eps > 0:
        grad2_rho = torch.autograd.grad(grad_rho[:, 0].sum(), xt, create_graph=True)[0]
        d2rho_dx2 = grad2_rho[:, 0] / x_max ** 2
        return continuity - diffusion_eps * d2rho_dx2
    else:
        return continuity


def train_pinn(
    trajectories: pl.DataFrame,
    grid: DiscretizationGrid,
    flow: numpy_implem.FlowFunction,
    conf: PINNConfig = PINNConfig(),
    device: str = "cpu",
) -> PINNResult:
    torch_flow = torch_implem_from_numpy(flow, frozen=not conf.learn_flow).to(device)
    is_arz = isinstance(torch_flow, ARZFlowBase)

    net = _make_network().to(device)
    params = list(net.parameters()) + (list(torch_flow.parameters()) if conf.learn_flow else [])
    adam = torch.optim.Adam(params, lr=conf.lr)
    lbfgs = torch.optim.LBFGS(params, lr=1.0, max_iter=20, history_size=50)

    # Normalize coordinates to [0, 1]
    x_max = grid.n_cells * grid.dx_meters
    t_max = grid.n_timesteps * grid.dt_seconds

    # Scale factors from frozen flow (O(1) normalization)
    rho_max: float = getattr(torch_flow, "rho_max").item()
    v_max: float = getattr(torch_flow, "v_max").item()

    # Observation data
    obs_x = torch.tensor(trajectories["x_meters"].to_numpy(), dtype=torch.float32, device=device) / x_max
    obs_t = torch.tensor(trajectories["time_seconds"].to_numpy(), dtype=torch.float32, device=device) / t_max
    obs_v = torch.tensor(trajectories["velocity"].to_numpy(), dtype=torch.float32, device=device)
    obs_xt = torch.stack([obs_x, obs_t], dim=1)

    wandb.init(project=conf.wandb_project, config={
        "n_collocation": conf.n_collocation,
        "epochs": conf.epochs,
        "lr": conf.lr,
        "observation_weight": conf.observation_weight,
        "physics_weight": conf.physics_weight,
        "flow": flow.__class__.__name__,
        "grid_dx": grid.dx_meters,
        "grid_dt": grid.dt_seconds,
    })

    _zero = torch.zeros(1, device=device)
    # initialise with a constant field so we always have a clean fallback
    with torch.no_grad():
        v_last, rho_last = _evaluate_on_grid(net, grid, x_max, t_max, device)
    best_loss = float("inf")
    for epoch in range(conf.epochs):
        optimizer = adam if epoch < conf.n_epochs_adam else lbfgs
        # L-BFGS requires a closure; also used for Adam for consistency
        xt_coll = torch.rand(conf.n_collocation, 2, device=device)
        loss_obs = _zero
        loss_continuity = _zero
        loss_momentum = _zero

        def closure() -> torch.Tensor:
            nonlocal loss_obs, loss_continuity, loss_momentum
            optimizer.zero_grad()

            out_obs = net(obs_xt)
            v_pred_obs = torch.nn.functional.softplus(out_obs[:, 1])
            loss_obs = torch.mean((v_pred_obs - obs_v) ** 2) / v_max ** 2

            if is_arz:
                arz_flow: ARZFlowBase = torch_flow  # type: ignore[assignment]
                res_cont, res_mom = _arz_residual(net, xt_coll, arz_flow, x_max, t_max)
                loss_continuity = torch.mean(res_cont ** 2) / (rho_max) ** 2
                loss_momentum = torch.mean(res_mom ** 2) / (v_max) ** 2
            else:
                res = _lwr_residual(net, xt_coll, torch_flow, x_max, t_max)
                loss_continuity = torch.mean(res ** 2) / (rho_max) ** 2
                loss_momentum = _zero

            total = conf.observation_weight * loss_obs + conf.physics_weight * (loss_continuity + loss_momentum)
            total.backward()
            return total

        optimizer.step(closure)

        total_loss = conf.observation_weight * loss_obs + conf.physics_weight * (loss_continuity + loss_momentum)
        best_loss = total_loss.item() if epoch == 0 else min(best_loss, total_loss.item())
        if torch.isnan(total_loss) or (epoch >= conf.n_epochs_adam and total_loss > 2*best_loss):
            print(f"Exploding loss at epoch {epoch + 1}, stopping early.")
            break

        log_dict: dict = {
            "loss": total_loss.item(),
            "loss_obs": (conf.observation_weight * loss_obs.item()),
            "loss_momentum": (conf.physics_weight * loss_momentum.item()) if is_arz else 0,
            "loss_continuity": (conf.physics_weight * loss_continuity.item()),
            "epoch": epoch + 1,
        }
        if (epoch + 1) % (conf.log_every if epoch < conf.n_epochs_adam else conf.log_every // 10) == 0:
            with torch.no_grad():
                v_last, rho_last = _evaluate_on_grid(net, grid, x_max, t_max, device)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].matshow(v_last.T, aspect="auto", origin="lower", cmap="turbo_r", vmin=0, vmax=35)
            axes[0].set_title("Velocity")
            axes[1].matshow(rho_last.T, aspect="auto", origin="lower", cmap="turbo")
            axes[1].set_title("Density")
            plt.tight_layout()
            log_dict["fields"] = wandb.Image(fig)
            plt.close(fig)
        wandb.log(log_dict)

    wandb.finish()

    return PINNResult(network=net, velocity_hat=v_last, rho_hat=rho_last, flow=torch_flow)


def _evaluate_on_grid(
    net: nn.Module,
    grid: DiscretizationGrid,
    x_max: float,
    t_max: float,
    device: str,
) -> tuple[NDArray, NDArray]:
    t_coords = (torch.arange(grid.n_timesteps, device=device) + 0.5) * grid.dt_seconds / t_max
    x_coords = (torch.arange(grid.n_cells, device=device) + 0.5) * grid.dx_meters / x_max
    tt, xx = torch.meshgrid(t_coords, x_coords, indexing="ij")
    xt = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=1)
    out = net(xt)
    rho = torch.nn.functional.softplus(out[:, 0]).reshape(grid.n_timesteps, grid.n_cells)
    v = torch.nn.functional.softplus(out[:, 1]).reshape(grid.n_timesteps, grid.n_cells)
    return v.cpu().numpy(), rho.cpu().numpy()
