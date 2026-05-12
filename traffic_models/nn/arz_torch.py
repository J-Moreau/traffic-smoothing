from abc import ABC
from typing import Literal

import torch
import torch.nn as nn

from traffic_models.nn.arz_high_res import rp1, weno_reconstruction
from traffic_models.sim import DiscretizationGrid


class ARZFlowBase(nn.Module, ABC):
    @property
    def v_max(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def rho_max(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def tau(self) -> torch.Tensor:
        raise NotImplementedError

    def h(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def d_hes(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rho_dh(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def rho_eq(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def v_eq(self, rho: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def q(self, rho: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return rho * (v + self.h(rho))

    def v(self, rho: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return (q / rho.clamp(min=1e-6) - self.h(rho)).clamp(
            min=1e-6, max=float(self.v_max) - 1e-6
        )

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        rho = U[..., 0, :]
        q = U[..., 1, :]
        return torch.stack(
            [q - rho * self.h(rho), q * q / rho.clamp(min=1e-6) - q * self.h(rho)],
            dim=-2,
        )

    def source_term(self, U: torch.Tensor) -> torch.Tensor:
        rho = U[..., 0, :]
        q = U[..., 1, :]
        return torch.stack(
            [
                torch.zeros_like(rho),
                (rho * self.v_eq(rho) + rho * self.h(rho) - q) / self.tau,
            ],
            dim=-2,
        )

    def lambdas(self, U: torch.Tensor) -> torch.Tensor:
        rho = U[..., 0, :]
        q = U[..., 1, :]
        v = self.v(rho, q)
        rho_dh = self.rho_dh(rho)
        lambda1 = v - rho_dh
        lambda2 = v
        return torch.stack([lambda1, lambda2], dim=-2)


class ARZFlow_power_law(ARZFlowBase):
    def __init__(
        self,
        v_max_init: float,
        rho_max_init: float,
        gamma: float = 1.0,
        tau: float = 5.0,
    ):
        super().__init__()
        self.log_v_max = nn.Parameter(
            torch.tensor(v_max_init, dtype=torch.float32).log()
        )
        self.log_rho_max = nn.Parameter(
            torch.tensor(rho_max_init, dtype=torch.float32).log()
        )
        self.log_gamma = nn.Parameter(
            torch.tensor(gamma, dtype=torch.float32).log()
        ).requires_grad_(False)
        # self.log_gamma = torch.tensor(gamma, dtype=torch.float32, requires_grad=False).log()
        self.log_tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32).log())

    @property
    def v_max(self) -> torch.Tensor:
        return self.log_v_max.exp()

    @property
    def rho_max(self) -> torch.Tensor:
        return self.log_rho_max.exp()

    @property
    def gamma(self) -> torch.Tensor:
        return self.log_gamma.exp()

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def h(self, rho: torch.Tensor) -> torch.Tensor:
        return self.v_max * (rho / self.rho_max).clamp(min=0).pow(self.gamma)

    def d_hes(self, rho: torch.Tensor) -> torch.Tensor:
        return (
            self.v_max
            * self.gamma
            / self.rho_max
            * (rho / self.rho_max).clamp(min=0).pow(self.gamma - 1)
        )

    def rho_dh(self, rho: torch.Tensor) -> torch.Tensor:
        return (
            self.v_max * self.gamma * (rho / self.rho_max).clamp(min=0).pow(self.gamma)
        )

    def rho_eq(self, v: torch.Tensor) -> torch.Tensor:
        return self.rho_max * ((self.v_max - v).clamp(min=0) / self.v_max).pow(
            1 / self.gamma
        )

    def v_eq(self, rho: torch.Tensor) -> torch.Tensor:
        return self.v_max * (1 - (rho / self.rho_max).clamp(min=0).pow(self.gamma))

    def lambdas(self, U: torch.Tensor) -> torch.Tensor:
        rho = U[..., 0, :]
        q = U[..., 1, :]
        v = self.v(rho, q)
        lambda1 = v - self.rho_dh(rho)
        lambda2 = v
        return torch.stack([lambda1, lambda2], dim=-2)


class ARZFlow_exponential(ARZFlowBase):
    def __init__(
        self,
        v_max_init: float,
        rho_max_init: float,
        c_jam: float = -4.0,
        tau: float = 5.0,
    ):
        super().__init__()
        self.log_v_max = nn.Parameter(
            torch.tensor(v_max_init, dtype=torch.float32).log()
        )
        self.log_rho_max = nn.Parameter(
            torch.tensor(rho_max_init, dtype=torch.float32).log()
        )
        assert c_jam < 0, "c_jam should be negative"
        self.log_c_jam = nn.Parameter(torch.tensor(-c_jam, dtype=torch.float32).log())
        self.log_tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32).log())

    @property
    def v_max(self) -> torch.Tensor:
        return self.log_v_max.exp()

    @property
    def rho_max(self) -> torch.Tensor:
        return self.log_rho_max.exp()

    @property
    def c_jam(self) -> torch.Tensor:
        return -self.log_c_jam.exp()

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def v_eq(self, rho: torch.Tensor) -> torch.Tensor:
        inner = self.c_jam / self.v_max * (1 - self.rho_max / rho.clamp(min=1e-6))
        return self.v_max * (1 - torch.exp(1 - torch.exp(inner.clamp(max=10))))

    def rho_eq(self, v: torch.Tensor) -> torch.Tensor:
        v_ratio = (1 - v / self.v_max).clamp(min=1e-6)
        inner = torch.log(torch.clamp(1 - torch.log(v_ratio), min=1e-6))
        return self.rho_max / (1 - self.v_max / self.c_jam * inner).clamp(min=1e-6)

    def dv_drho(self, rho: torch.Tensor) -> torch.Tensor:
        c_jam, v_max, rho_max = self.c_jam, self.v_max, self.rho_max
        ratio = c_jam / v_max * (1 - rho_max / rho.clamp(min=1e-6))
        return (c_jam * rho_max / rho.clamp(min=1e-6).pow(2)) * torch.exp(
            1 - torch.exp(ratio.clamp(max=10)) + ratio.clamp(max=10)
        )

    def h(self, rho: torch.Tensor) -> torch.Tensor:
        rho_zero = torch.full_like(rho, 1e-6)
        return self.v_eq(rho_zero) - self.v_eq(rho)

    def rho_dh(self, rho: torch.Tensor) -> torch.Tensor:
        return -rho * self.dv_drho(rho)


class ARZFlowQuadraticLinear(ARZFlowBase):
    def __init__(
        self,
        v_max_init: float,
        rho_max_init: float,
        rho_c: float = 0.05,
        tau: float = 5.0,
    ):
        super().__init__()
        self.log_v_max = nn.Parameter(
            torch.tensor(v_max_init, dtype=torch.float32).log(),
            requires_grad=False
        )
        self.log_rho_max = nn.Parameter(
            torch.tensor(rho_max_init, dtype=torch.float32).log()
        )
        self.log_rho_c = nn.Parameter(torch.tensor(rho_c, dtype=torch.float32).log())
        self.log_tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32).log())

    @property
    def v_max(self) -> torch.Tensor:
        return self.log_v_max.exp()

    @property
    def rho_max(self) -> torch.Tensor:
        return self.log_rho_max.exp()

    @property
    def rho_c(self) -> torch.Tensor:
        return self.log_rho_c.exp()

    @property
    def v_c(self) -> torch.Tensor:
        return self.v_max * (1 - self.rho_c / self.rho_max)

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def rho_eq(self, v: torch.Tensor) -> torch.Tensor:
        """Compute density from velocity."""
        v_c = self.v_c
        free_flow = self.rho_max * (1 - v / self.v_max) * (v > v_c)
        congestion = (
            self.rho_max / (1 + v / self.v_max * self.rho_max / self.rho_c) * (v <= v_c)
        )
        return free_flow + congestion

    def v_eq(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute velocity from density."""
        eps = 1e-8
        free_flow = self.v_max * (1 - rho / self.rho_max) * (rho < self.rho_c)
        congestion = (
            self.rho_c
            * self.v_max
            * (1 / torch.clamp(rho, min=eps) - 1 / self.rho_max)
            * (rho >= self.rho_c)
        )
        return free_flow + congestion

    def dv_drho(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of velocity with respect to density."""
        eps = 1e-8
        free_flow = -self.v_max / self.rho_max * (rho < self.rho_c)
        congestion = (
            -self.rho_c
            * self.v_max
            / torch.clamp(rho.pow(2), min=eps)
            * (rho >= self.rho_c)
        )
        return free_flow + congestion

    def h(self, rho: torch.Tensor) -> torch.Tensor:
        rho_zero = torch.full_like(rho, 1e-6)
        return self.v_eq(rho_zero) - self.v_eq(rho)

    def rho_dh(self, rho: torch.Tensor) -> torch.Tensor:
        return -rho * self.dv_drho(rho)


class ARZFlowCongested(ARZFlowBase):
    """
    Designed so that rho_v_eq is constant in free flow and greenshields in congestion
    """

    def __init__(self, v_max_init: float, rho_max_init: float, tau: float = 5.0):
        super().__init__()
        self.log_v_max = nn.Parameter(
            torch.tensor(v_max_init, dtype=torch.float32).log()
        )
        self.log_rho_max = nn.Parameter(
            torch.tensor(rho_max_init, dtype=torch.float32).log()
        )
        self.log_tau = nn.Parameter(torch.tensor(tau, dtype=torch.float32).log())

    @property
    def v_max(self) -> torch.Tensor:
        return self.log_v_max.exp()

    @property
    def rho_max(self) -> torch.Tensor:
        return self.log_rho_max.exp()

    @property
    def rho_c(self) -> torch.Tensor:
        return self.rho_max * 0.5  # fixed critical density at half of max density

    @property
    def v_c(self) -> torch.Tensor:
        return self.v_max * 0.5

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def rho_eq(self, v: torch.Tensor) -> torch.Tensor:
        """Compute density from velocity."""
        v_c = self.v_c
        free_flow = 1 / 4 * self.rho_max * self.v_max / v.clamp(v_c) * (v > v_c)
        congestion = self.rho_max * (1 - v / self.v_max) * (v <= v_c)
        return free_flow + congestion

    def v_eq(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute velocity from density."""
        free_flow = (
            1
            / 4
            * self.v_max
            * self.rho_max
            / rho.clamp(self.rho_max / 4)
            * (rho < self.rho_c)
        )
        congestion = self.v_max * (1 - rho / self.rho_max) * (rho >= self.rho_c)
        return free_flow + congestion

    def dv_drho(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of velocity with respect to density."""
        free_flow = (
            -1
            / 4
            * self.v_max
            * self.rho_max
            / rho.clamp(self.rho_max / 4).pow(2)
            * (rho < self.rho_c)
        )
        congestion = -self.v_max / self.rho_max * (rho >= self.rho_c)
        return free_flow + congestion

    def h(self, rho: torch.Tensor) -> torch.Tensor:
        return (self.v_max - self.v_eq(rho)).clamp(min=0)

    def rho_dh(self, rho: torch.Tensor) -> torch.Tensor:
        return -rho * self.dv_drho(rho)

    def lambdas(self, U: torch.Tensor) -> torch.Tensor:
        rho = U[..., 0, :]
        q = U[..., 1, :]
        v = self.v(rho, q)
        lambda1 = v - self.rho_dh(rho)
        lambda2 = v
        return torch.stack([lambda1, lambda2], dim=-2).clamp(
            max=0
        )  # only propagate backwards

    def forward(self, U: torch.Tensor) -> torch.Tensor:
        rho = U[..., 0, :]
        q = U[..., 1, :]
        return torch.stack(
            [q - rho * self.h(rho), q * q / rho.clamp(min=1e-6) - q * self.h(rho)],
            dim=-2,
        )


class ARZFlowPiecewiseQuadratic(ARZFlowBase):
    v_max_init: float
    rho_max_init: float
    rho_c_init: float
    Q_max_init: float
    tau_init: float = 5.0

    def __init__(
        self,
        v_max_init: float,
        rho_max_init: float,
        rho_c_init: float,
        Q_max_init: float,
        tau_init: float = 5.0,
    ):
        super().__init__()
        self.log_v_max = torch.tensor(v_max_init, dtype=torch.float32).log()
        self.log_rho_max = nn.Parameter(
            torch.tensor(rho_max_init, dtype=torch.float32).log()
        )
        self.log_rho_c = nn.Parameter(
            torch.tensor(rho_c_init, dtype=torch.float32).log()
        )
        self.log_Q_max = nn.Parameter(
            torch.tensor(Q_max_init, dtype=torch.float32).log()
        )
        self.log_tau = nn.Parameter(torch.tensor(tau_init, dtype=torch.float32).log())

    @property
    def v_max(self) -> torch.Tensor:
        return self.log_v_max.exp()

    @property
    def rho_max(self) -> torch.Tensor:
        return self.log_rho_max.exp()

    @property
    def rho_c(self) -> torch.Tensor:
        return self.log_rho_c.exp()

    @property
    def Q_max(self) -> torch.Tensor:
        return self.log_Q_max.exp()

    @property
    def v_c(self) -> torch.Tensor:
        return self.Q_max / self.rho_c

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp()

    def rho_eq(self, v: torch.Tensor) -> torch.Tensor:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        free_flow = rho_1 * (1 - v / self.v_max) * (v > self.v_c)
        # not the true inverse but still fine for our purposes, and easier to compute
        congestion = (
            self.rho_max
            / (1 + v / self.v_max * self.rho_max / self.rho_c)
            * (v <= self.v_c)
        )
        return free_flow + congestion

    def v_eq(self, rho: torch.Tensor) -> torch.Tensor:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        free_flow = self.v_max * (1 - rho / rho_1)
        congestion = (
            self.Q_max
            * (rho - self.rho_max)
            / (self.rho_c - self.rho_max)
            * (2 * self.rho_c - self.rho_max - rho)
            / (self.rho_c - self.rho_max)
            / torch.where(rho >= self.rho_c, rho, 1.0)  # avoid division by zero
        )
        return free_flow * (rho < self.rho_c) + congestion * (rho >= self.rho_c)

    def dv_drho(self, rho: torch.Tensor) -> torch.Tensor:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        free_flow = -self.v_max / rho_1
        congestion = (
            self.Q_max
            * (2 * self.rho_max * self.rho_c - self.rho_max**2 - rho**2)
            / (self.rho_c - self.rho_max) ** 2
            / torch.where(rho >= self.rho_c, rho**2, 1.0)
        )  # avoid division by zero

        return free_flow * (rho < self.rho_c) + congestion * (rho >= self.rho_c)

    def h(self, rho: torch.Tensor) -> torch.Tensor:
        return self.v_max - self.v_eq(rho)

    def rho_dh(self, rho: torch.Tensor) -> torch.Tensor:
        return -rho * self.dv_drho(rho)


ARZFlow = (
    ARZFlow_power_law
    | ARZFlow_exponential
    | ARZFlowQuadraticLinear
    | ARZFlowCongested
    | ARZFlowPiecewiseQuadratic
)


def arz_rusanov_step(
    U: torch.Tensor,
    flow: ARZFlow,
    dt: float,
    dx: float,
    boundary: Literal["edge", "wrap"] = "edge",
) -> torch.Tensor:
    f_U = flow(U)
    speed = flow.lambdas(U).abs().max(dim=-2).values
    pad_mode = "replicate" if boundary == "edge" else "circular"
    speed_pad = torch.nn.functional.pad(
        speed.unsqueeze(0), (1, 1), mode=pad_mode
    ).squeeze(0)
    alpha = torch.maximum(speed_pad[..., 1:], speed_pad[..., :-1]).unsqueeze(-2)

    f_U_pad = torch.nn.functional.pad(f_U, (1, 1), mode=pad_mode)
    U_pad = torch.nn.functional.pad(U, (1, 1), mode=pad_mode)
    U_i = U_pad[..., :-1]
    U_ip1 = U_pad[..., 1:]
    f_U_i = f_U_pad[..., :-1]
    f_U_ip1 = f_U_pad[..., 1:]

    F_iphalf = 0.5 * (f_U_i + f_U_ip1) - 0.5 * alpha * (U_ip1 - U_i)
    U_next = U - dt / dx * (F_iphalf[..., 1:] - F_iphalf[..., :-1])
    return U_next + dt * flow.source_term(U)


def arz_lax_friedrichs_ctm_step(
    U: torch.Tensor,
    flow: ARZFlow,
    dt: float,
    dx: float,
    ramp_idx: torch.Tensor,
    flux_ratio: torch.Tensor,
    boundary: Literal["edge", "wrap"] = "edge",
    solver: Literal["lax_friedrichs", "rusanov"] = "rusanov",
) -> torch.Tensor:
    assert solver in ["lax_friedrichs", "rusanov"], (
        "solver must be either 'lax_friedrichs' or 'rusanov'"
    )
    f_U = flow(U)
    pad_mode = "replicate" if boundary == "edge" else "circular"
    f_U_pad = torch.nn.functional.pad(f_U, (1, 1), mode=pad_mode)
    U_pad = torch.nn.functional.pad(U, (1, 1), mode=pad_mode)
    if solver == "rusanov":
        speed = flow.lambdas(U).abs().max(dim=-2).values
        speed_pad = torch.nn.functional.pad(
            speed.unsqueeze(0), (1, 1), mode=pad_mode
        ).squeeze(0)
        alpha = torch.maximum(speed_pad[..., 1:], speed_pad[..., :-1]).unsqueeze(-2)
        # alpha = torch.where(torch.maximum(U_pad[..., 0:1, :-1], U_pad[..., 0:1, 1:]) < flow.rho_max/4, dt/dx, alpha)  # increase diffusion if low density to account for uncertainty
    else:
        alpha = dt / dx
    U_i = U_pad[..., :-1]
    U_ip1 = U_pad[..., 1:]
    f_U_i = f_U_pad[..., :-1]
    f_U_ip1 = f_U_pad[..., 1:]

    F_iphalf = 0.5 * (f_U_i + f_U_ip1) - 0.5 * alpha * (U_ip1 - U_i)

    F_right = F_iphalf[..., 1:].clone()
    F_left = F_iphalf[..., :-1].clone()

    # simple on/off ramp modelling
    # the influx of cell i+1 is multiplied by a ratio
    # ratio should be < 1 for off-ramps and > 1 for on-ramps, and can be learned
    r = U[..., 0, ramp_idx - 1]
    q = U[..., 1, ramp_idx - 1]
    F_left[..., 0, ramp_idx] = (
        F_right[..., 0, ramp_idx - 1] * flux_ratio
    )  # a ratio of the vehicles enter/exit the motorway

    F_left[..., 1, ramp_idx] = F_left[..., 0, ramp_idx] * (flow.v(r, q) + flow.h(r))

    U_next = U - dt / dx * (F_right - F_left)
    return U_next + dt * flow.source_term(U)


def arz_lax_friedrichs_step(
    U: torch.Tensor, flow: ARZFlow, dt: float, dx: float
) -> torch.Tensor:
    f_U = flow(U)
    alpha = dt / (2 * dx)

    U_next = 0.5 * (torch.roll(U, 1, -1) + torch.roll(U, -1, -1)) - alpha * (
        torch.roll(f_U, -1, -1) - torch.roll(f_U, 1, -1)
    )
    # mirror boundary conditions
    U_next[..., 0] = 0.5 * (U[..., 0] + U[..., 1]) - alpha * (f_U[..., 1] - f_U[..., 0])
    U_next[..., -1] = 0.5 * (U[..., -2] + U[..., -1]) - alpha * (
        f_U[..., -1] - f_U[..., -2]
    )

    return U_next + dt * flow.source_term(U)


def arz_solver_batch(
    U: torch.Tensor,
    flow: ARZFlow,
    grid: DiscretizationGrid,
    ramp_idx: torch.Tensor,
    flux_ratio: torch.Tensor,
    solver: Literal["godunov", "lax_friedrichs", "rusanov"] = "lax_friedrichs",
) -> torch.Tensor:
    if solver == "lax_friedrichs" or solver == "rusanov":
        return arz_lax_friedrichs_ctm_step(
            U,
            flow,
            grid.dt_seconds,
            grid.dx_meters,
            ramp_idx=ramp_idx,
            flux_ratio=flux_ratio,
            solver=solver,
        )
    if solver == "godunov":
        return arz_godunov_step_batch(
            U, flow, grid.dt_seconds, grid.dx_meters, boundary="mirror"
        )
    raise ValueError(
        f"Unknown solver {solver}, only 'lax_friedrichs', 'godunov' and 'rusanov' are supported for ARZ"
    )


def arz_solver_rollout(
    U_init: torch.Tensor,
    flow: ARZFlow,
    grid: DiscretizationGrid,
    ramp_idx: torch.Tensor,
    flux_ratio: torch.Tensor,
    source_term: torch.Tensor,
    solver: Literal["lax_friedrichs", "rusanov", "godunov"] = "rusanov",
) -> torch.Tensor:
    U_t = U_init.clone()
    output = [U_t]
    for _ in range(grid.n_timesteps - 1):
        if solver == "lax_friedrichs" or solver == "rusanov":
            output.append(
                arz_lax_friedrichs_ctm_step(
                    output[-1],
                    flow,
                    grid.dt_seconds,
                    grid.dx_meters,
                    ramp_idx=ramp_idx,
                    flux_ratio=flux_ratio,
                    solver=solver,
                ).clone()
            )
        elif solver == "godunov":
            output.append(
                arz_godunov_step_batch(
                    output[-1], flow, grid.dt_seconds, grid.dx_meters, boundary="mirror"
                ).clone()
            )
        output[-1][0, :].clamp_(
            min=1e-6, max=float(flow.rho_max) - 1e-6
        )  # clamp density to physical bounds
        output[-1][1, :].clamp_(
            min=1e-6, max=float(flow.v_max * flow.rho_max) - 1e-6
        )  # clamp q to physical bounds
        output[-1][0, ramp_idx] += (
            source_term * grid.dt_seconds
        )  # add source term to density
    return torch.stack(output, dim=0)


def arz_godunov_rollout(
    rho: torch.Tensor,
    q: torch.Tensor | None,
    flow: ARZFlow,
    grid: DiscretizationGrid,
    boundary: Literal["mirror", "periodic"] = "mirror",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eps = 1e-6
    Nt = grid.n_timesteps
    Nx = grid.n_cells
    v_max = float(flow.v_max)

    if q is None:
        v = flow.v_eq(rho)
        w = v + flow.h(rho)
        q = rho * w

    rho_hist = torch.zeros((Nt, Nx), dtype=rho.dtype, device=rho.device)
    v_hist = torch.zeros((Nt, Nx), dtype=rho.dtype, device=rho.device)
    q_hist = torch.zeros((Nt, Nx), dtype=rho.dtype, device=rho.device)
    for n in range(Nt):
        U = torch.stack([rho, q], dim=-2)
        arz_godunov_step_batch(U, flow, grid.dt_seconds, grid.dx_meters, boundary)
        rho, q = U[0], U[1]
        v = q / rho.clamp(min=eps) - flow.h(rho)
        v = v.clamp(min=eps, max=v_max - eps)
        rho_hist[n] = rho
        q_hist[n] = q
        v_hist[n] = v

    return rho_hist, q_hist, v_hist


def arz_godunov_step_batch(
    U: torch.Tensor,
    flow: ARZFlow,
    dt: float,
    dx: float,
    boundary: Literal["mirror", "periodic"] = "mirror",
):
    """Sharpclaw-like Godunov step with WENO reconstruction and second-order correction, translated from sharpclaw code by David Ketcheson"""
    eps = 1e-6
    h = flow.h
    rho_max = float(flow.rho_max)
    v_max = float(flow.v_max)
    rho_dhes = flow.rho_dh
    source_term = flow.source_term
    bc = 2  # Number of ghost cells on each side
    pad_mode = "replicate" if boundary == "mirror" else "circular"
    if U.ndim == 2:
        # squeezing to also handle non-batched input of shape (2, n_cells)
        U_pad = torch.nn.functional.pad(U.unsqueeze(0), (2, 2), mode=pad_mode).squeeze(
            0
        )
    else:
        U_pad = torch.nn.functional.pad(U, (2, 2), mode=pad_mode)

    # U = torch.nn.functional.conv1d(
    #     torch.nn.functional.pad(U, (2, 2), mode=pad_mode)
    #     , torch.ones((1,*U.shape[-2:-1],5), device=U.device)/5, padding="same")  # smooth initial density

    Up_imhalf, Um_iphalf = weno_reconstruction(U_pad)
    Up_imhalf = torch.nn.functional.pad(Up_imhalf, (bc, bc), mode=pad_mode)
    Um_iphalf = torch.nn.functional.pad(Um_iphalf, (bc, bc), mode=pad_mode)

    # Inter-cell fluxes
    _, _, amdq, apdq = rp1(Up_imhalf, Um_iphalf, h, rho_dhes)

    # Intra-cell fluxes (high resolution terms)
    _, _, amdq2, apdq2 = rp1(Um_iphalf, torch.roll(Up_imhalf, -1, dims=-1), h, rho_dhes)

    dU = -1 / dx * (apdq + torch.roll(amdq, -1, dims=-1) + apdq2 + amdq2)
    dU = dU + source_term(U_pad)

    U_next = U_pad + dU * dt
    # Clip to physical bounds
    U_next[..., 0, :].clamp_(min=eps, max=rho_max - eps)
    U_next[..., 1, :].clamp_(min=eps, max=2 * rho_max * v_max - eps)
    return U_next[..., bc:-bc]
