import torch

from traffic_models.nn.flows_torch import GreenshieldsFlow, TriangularFlow


def lax_friedrichs_step(
    flow: GreenshieldsFlow|TriangularFlow, rho: torch.Tensor, dt_seconds: float, dx_meters: float
) -> torch.Tensor:
    # 1/2 (rho_{k+1} + rho_{k-1})
    # - dt/2dx (Q(rho_{k+1}) - Q(rho_{k-1}))
    f_rho = flow(rho)
    rho_next = 0.5 * (
        torch.roll(rho, 1, -1) + torch.roll(rho, -1, -1)
    ) - dt_seconds / (2 * dx_meters) * (
        torch.roll(f_rho, -1, -1) - torch.roll(f_rho, 1, -1)
    )
    # at the boundaries we assume mirror rho values for ghost cells
    rho_next[..., 0] = 0.5 * (rho[..., 1] + rho[..., 0]) - dt_seconds / (2 * dx_meters) * (
        f_rho[..., 1] - f_rho[..., 0]
    )
    rho_next[..., -1] = 0.5 * (rho[..., -2] + rho[..., -1]) - dt_seconds / (
        2 * dx_meters
    ) * (f_rho[..., -1] - f_rho[..., -2])
    return rho_next


def lax_friedrichs_jacobian(
    rho: torch.Tensor, flow: GreenshieldsFlow, dt_seconds: float, dx_meters: float
) -> torch.Tensor:
    return torch.autograd.functional.jacobian(
        lambda r: lax_friedrichs_step(flow, r, dt_seconds, dx_meters),
        rho,
        create_graph=True,
    )
