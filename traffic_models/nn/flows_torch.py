import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parametrize

import traffic_models.flows as numpy_implem
from traffic_models.nn.arz_torch import (
    ARZFlow,
    ARZFlow_exponential,
    ARZFlow_power_law,
    ARZFlowCongested,
    ARZFlowPiecewiseQuadratic,
    ARZFlowQuadraticLinear,
)


class TriangularFlow(torch.nn.Module):
    def __init__(self, rho_c_init: float, rho_max_init: float, Q_c_init: float, rho_free_flow_init: float):
        super().__init__()
        self.log_rho_c = torch.nn.Parameter(
            torch.tensor(np.log(rho_c_init), dtype=torch.float32)
        )
        self.log_rho_max = torch.nn.Parameter(
            torch.tensor(np.log(rho_max_init), dtype=torch.float32)
        )
        self.log_Q_c = torch.nn.Parameter(
            torch.tensor(np.log(Q_c_init), dtype=torch.float32)
        )
        self.log_rho_free_flow = torch.nn.Parameter(
            torch.tensor(np.log(rho_free_flow_init), dtype=torch.float32)
        )

    @property
    def v_max(self) -> torch.Tensor:
        return torch.exp(self.log_Q_c - self.log_rho_c)
    
    def forward(self, rho: torch.Tensor):
        rho_c = torch.exp(self.log_rho_c)
        Q_c = torch.exp(self.log_Q_c)
        rho_max = torch.exp(self.log_rho_max)

        under_critical = (Q_c * rho / rho_c) * (rho < rho_c)
        over_critical = (Q_c * (rho_max - rho) / (rho_max - rho_c)) * (rho >= rho_c)
        Q = under_critical + over_critical
        return Q
    
    def velocity_from_density(self, rho: torch.Tensor) -> torch.Tensor:
        """
        Inverse function: velocity from density
        """
        rho_c = torch.exp(self.log_rho_c)
        Q_c = torch.exp(self.log_Q_c)
        rho_max = torch.exp(self.log_rho_max)
        v_max = Q_c/rho_c

        under_critical = v_max * (rho < rho_c)
        over_critical = Q_c * (rho_max/rho - 1) / (rho_max - rho_c) * (rho >= rho_c)
        return under_critical + over_critical
    
    def density_from_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """
        Inverse function: density from velocity
        This relation is not bijective for free flow conditions v >= v_max
        so we return an arbitrary constant free-flow density rho_free_flow
        In "Incorporation of Lagrangian Measurements in freeway TSE" (2010)
        Herrera et Bayen use:   rho_free_flow = 5/6 * rho_c
        """
        rho_c = torch.exp(self.log_rho_c)
        Q_c = torch.exp(self.log_Q_c)
        rho_max = torch.exp(self.log_rho_max)
        rho_free_flow = torch.exp(self.log_rho_free_flow)
        v_max = Q_c/rho_c

        # congestion flow slope
        w = Q_c  / (rho_max - rho_c)

        congestion =  rho_max / (1 + v/w)* (v <= v_max)
        free_flow = rho_free_flow * (v > v_max)
        return free_flow + congestion

    @property
    def rho_c(self) -> torch.Tensor:
        """Critical density"""
        return torch.exp(self.log_rho_c)
    
    @property
    def rho_max(self) -> torch.Tensor:
        return torch.exp(self.log_rho_max)


class GreenshieldsFlow(torch.nn.Module):
    """Density-based Greenshields flow model Q(rho) = rho * v_max * (1 - rho / rho_max)"""
    def __init__(self, v_max_init: float, rho_max_init: float):
        super().__init__()
        self.log_v_max = torch.nn.Parameter(
            torch.tensor(np.log(v_max_init), dtype=torch.float32)
        )
        self.log_rho_max = torch.nn.Parameter(
            torch.tensor(np.log(rho_max_init), dtype=torch.float32)
        )

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        return rho * torch.exp(self.log_v_max) * (1 - rho / torch.exp(self.log_rho_max))

    def velocity_from_density(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute velocity from density using Greenshields model."""
        v_max = torch.exp(self.log_v_max)
        rho_max = torch.exp(self.log_rho_max)
        return v_max * (1 - rho / rho_max)

    def density_from_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Compute density from velocity using Greenshields model."""
        v_max = torch.exp(self.log_v_max)
        rho_max = torch.exp(self.log_rho_max)
        return rho_max * (1 - v / v_max)
    
    @property
    def rho_c(self) -> torch.Tensor:
        """Critical density"""
        return torch.exp(self.log_rho_max) / 2
    
    @property
    def rho_max(self) -> torch.Tensor:
        return torch.exp(self.log_rho_max)

    @property
    def v_max(self) -> torch.Tensor:
        return torch.exp(self.log_v_max)
    

class PiecewiseQuadraticFlow(torch.nn.Module):
    v_max_init: float
    rho_max_init: float
    rho_c_init: float
    Q_max_init: float
    
    def __init__(
        self,
        v_max_init: float,
        rho_max_init: float,
        rho_c_init: float,
        Q_max_init: float,
    ):
        super().__init__()
        self.log_v_max = torch.tensor(v_max_init, dtype=torch.float32).log()
        self.log_rho_max = nn.Parameter(
            torch.tensor(rho_max_init, dtype=torch.float32).log()
        )
        self.log_rho_c = nn.Parameter(torch.tensor(rho_c_init, dtype=torch.float32).log())
        self.log_Q_max = nn.Parameter(torch.tensor(Q_max_init, dtype=torch.float32).log())

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
        return self.Q_max/self.rho_c

    def density_from_velocity(self, v: torch.Tensor) -> torch.Tensor:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        free_flow = rho_1 * (1 - v / self.v_max) * (v > self.v_c)
        congestion = self.rho_max / (1 + v / self.v_max * self.rho_max/self.rho_c) * (v <= self.v_c)
        return free_flow + congestion

    # If we are symmetric around zero it doesnt make the inversion easier
    # Q = Q_max*(rho_max**2 - rho**2)/(rho_max**2 - rho_c**2) for rho >= rho_c
    # v = Q/rho = Q_max*(rho_max**2/rho - rho) / (rho_max**2 - rho_c**2)
    # 0 = Q_max*(rho_max**2 - rho**2) / (rho_max**2 - rho_c**2) - v*rho
    def velocity_from_density(self, rho: torch.Tensor) -> torch.Tensor:
        rho_1 = self.rho_c / (1 - self.Q_max / (self.v_max * self.rho_c))
        free_flow = self.v_max * (1 - rho / rho_1) 
        congestion = (self.Q_max * (rho - self.rho_max)/(self.rho_c - self.rho_max)
            * (self.rho_max + rho) / (self.rho_c + self.rho_max)
            / torch.where(rho >= self.rho_c, rho, 1.0) # avoid division by zero 
            )
        return free_flow * (rho < self.rho_c) + congestion * (rho >= self.rho_c)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        free_flow = self.v_max * rho * (1 - rho / self.rho_c) * (rho < self.rho_c)
        congestion = self.Q_max * (self.rho_max - rho)/(self.rho_max - self.rho_c) * (self.rho_max + rho) / (self.rho_c + self.rho_max) * (rho >= self.rho_c)
        return free_flow + congestion


class GreenshieldsVelocityFlow(torch.nn.Module):
    """Velocity-based Greenshields flow model Q(v) = v * (v - v_max)"""
    def __init__(self, v_max_init: float):
        super().__init__()
        self.log_v_max = torch.nn.Parameter(
            torch.tensor(np.log(v_max_init), dtype=torch.float32)
        )
        
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return v * (v-torch.exp(self.log_v_max))


class SoftplusParameterization(torch.nn.Module):
    def forward(self, W):
        return torch.nn.functional.softplus(W)
    
class ConvexVelocityFlow(nn.Module):
    def __init__(self, hidden_features: int, init_scale: float = 0.1):
        super().__init__()
        self.a = nn.Parameter(init_scale * torch.randn(hidden_features))
        self.b = nn.Parameter(init_scale * torch.randn(hidden_features))
        self.q_unconstrained = nn.Parameter(torch.tensor(0.0))

    @property
    def q(self):
        return torch.nn.functional.softplus(self.q_unconstrained) + 1e-8
    
    @property
    def v_c(self):
        return self.minimize_closed_form()

    def forward(self, x_unscaled):
        x = x_unscaled / 40 #v_max
        axb = x.unsqueeze(-1) * self.a + self.b  # (...,k)
        m = axb.max(dim=-1).values
        y = 0.5 * self.q * x**2 + m
        return y

    @torch.no_grad()
    def minimize_closed_form(self):
        a = self.a.detach()
        b = self.b.detach()
        q = self.q.detach()

        # --- All candidate optimal points (vectorized)
        xs = -a / q                       # shape (k,)

        # Evaluate all affine pieces at all xs: (k, k)
        vals = xs.unsqueeze(1) * a + b    # i.e. vals[i,j] = a_j*xs[i] + b_j

        # For each xs[i], which j maximizes a_j x + b_j?
        active = torch.argmax(vals, dim=1)  # shape (k,)

        # Keep only xs[i] where piece i is active
        mask = active == torch.arange(a.numel())
        xs_valid = xs[mask]

        # If nothing valid (rare), fall back to argmin over all candidates
        if xs_valid.numel() == 0:
            xs_valid = xs

        # Evaluate f(x) on valid candidates
        fvals = 0.5 * q * xs_valid**2 + torch.max(xs_valid.unsqueeze(1)*a + b, dim=1).values

        # Pick minimizer
        idx = torch.argmin(fvals)
        return xs_valid[idx]


class QuadraticLinearFlow(nn.Module):
    """
    PyTorch version of QuadraticLinearFlow
    
    From "A traffic model for velocity assimilation", D B Work et al 2010. Eq 10,11
    A flow function that is quadratic for low densities and linear for high densities.
    This allows the inversion of the velocity - density relation.
    """
    def __init__(
        self,
        v_max_init: float,
        rho_max_init: float,
        rho_c_init: float,
    ):
        super().__init__()
        self.log_v_max = torch.tensor(v_max_init, dtype=torch.float32).log()
        self.log_rho_max = nn.Parameter(
            torch.tensor(rho_max_init, dtype=torch.float32).log()
        )
        self.log_rho_c = nn.Parameter(torch.tensor(rho_c_init, dtype=torch.float32).log())

    @property
    def v_max(self) -> torch.Tensor:
        return torch.exp(self.log_v_max)

    @property
    def rho_max(self) -> torch.Tensor:
        return torch.exp(self.log_rho_max)

    @property
    def rho_c(self) -> torch.Tensor:
        return torch.exp(self.log_rho_c)

    @property
    def v_c(self) -> torch.Tensor:
        return self.v_max * (1 - self.rho_c / self.rho_max)

    def forward(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute flow from density using quadratic-linear model."""
        quad = self.v_max * rho * (1 - rho / self.rho_max) * (rho < self.rho_c)
        lin = self.rho_c * self.v_max * (1 - rho / self.rho_max) * (rho >= self.rho_c)
        return quad + lin

    def density_from_velocity(self, v: torch.Tensor) -> torch.Tensor:
        """Compute density from velocity."""
        v_c = self.v_c
        free_flow = self.rho_max * (1 - v / self.v_max) * (v > v_c)
        congestion = self.rho_max / (1 + v / self.v_max * self.rho_max / self.rho_c) * (v <= v_c)
        return free_flow + congestion

    def velocity_from_density(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute velocity from density."""
        free_flow = self.v_max * (1 - rho / self.rho_max) * (rho < self.rho_c)
        congestion = self.rho_c * self.v_max * (
            1 / torch.clamp(rho, min=self.rho_c) - 1 / self.rho_max
        ) * (rho >= self.rho_c)
        return free_flow + congestion


def torch_implem_from_numpy(flow: numpy_implem.FlowFunction, frozen: bool = True) -> GreenshieldsFlow|TriangularFlow|ARZFlow:
    if isinstance(flow, numpy_implem.GreenshieldsFlow):
        torch_flow = GreenshieldsFlow(flow.v_max, flow.rho_max)
    elif isinstance(flow, numpy_implem.TriangularFlow):
        torch_flow = TriangularFlow(rho_c_init=flow.rho_c, rho_max_init=flow.rho_max, Q_c_init=flow.Q_c, rho_free_flow_init=flow.rho_free_flow)
    elif isinstance(flow, numpy_implem.PiecewiseQuadraticFlow):
        torch_flow = PiecewiseQuadraticFlow(v_max_init=flow.v_max, rho_max_init=flow.rho_max, rho_c_init=flow.rho_c, Q_max_init=flow.Q_max)
    elif isinstance(flow, numpy_implem.QuadraticLinearFlow):
        torch_flow = QuadraticLinearFlow(v_max_init=flow.v_max, rho_max_init=flow.rho_max, rho_c_init=flow.rho_c)
    elif isinstance(flow, numpy_implem.ARZFlow):
        if isinstance(flow.flux_function, numpy_implem.PowerLawFlux):
            torch_flow = ARZFlow_power_law(v_max_init=flow.v_max, rho_max_init=flow.rho_max, gamma=flow.flux_function.gamma, tau=flow.tau)
        elif isinstance(flow.flux_function, numpy_implem.DelCastilloBenitezFlux):
            torch_flow = ARZFlow_exponential(v_max_init=flow.v_max, rho_max_init=flow.rho_max, c_jam=flow.flux_function.c_jam, tau=flow.tau)
        elif isinstance(flow.flux_function, numpy_implem.QuadraticLinearFlow):
            torch_flow = ARZFlowQuadraticLinear(v_max_init=flow.v_max, rho_max_init=flow.rho_max, rho_c=flow.flux_function.rho_c, tau=flow.tau)
        elif isinstance(flow.flux_function, numpy_implem.CongestedFlow):
            torch_flow = ARZFlowCongested(v_max_init=flow.v_max, rho_max_init=flow.rho_max, tau=flow.tau)
        elif isinstance(flow.flux_function, numpy_implem.PiecewiseQuadraticFlow):
            torch_flow = ARZFlowPiecewiseQuadratic(v_max_init=flow.v_max, rho_max_init=flow.rho_max, rho_c_init=flow.flux_function.rho_c, Q_max_init=flow.flux_function.Q_max, tau_init=flow.tau)
        else:
            raise ValueError(f"Unsupported ARZ flux function type: {type(flow.flux_function)}")
    else:
        raise ValueError(f"Unsupported flow type: {type(flow)}")
    for param in torch_flow.parameters():
        param.requires_grad = not frozen
    return torch_flow
