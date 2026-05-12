import numpy as np
import torch

import traffic_models.nn.flows_torch as flows_torch
from traffic_models.flows import ARZFlow, GreenshieldsFlow, TriangularFlow
from traffic_models.lax_friedrichs import (
    arz_gaussian_initial_condition,
    arz_rollout,
    lax_friedrichs_affine_update,
    lax_friedrichs_step,
    lax_friedrichs_step_batch,
)
from traffic_models.nn.lax_friedrichs_torch import (
    lax_friedrichs_step as lax_friedrichs_step_torch,
)
from traffic_models.sim import DiscretizationGrid


def test_lax_friedrichs_step_equivalence_between_np_and_torch():
    grid = DiscretizationGrid(
        dt_seconds=0.1,
        dx_meters=0.1,
        n_timesteps=5,
        n_cells=5,
    )

    # Triangular Flow
    t_flow = TriangularFlow(rho_c=0.5, Q_c=0.5, rho_max=1.0, rho_free_flow=0.2)
    rho = np.linspace(0, t_flow.rho_max, grid.n_cells)
    # Numpy version
    rho_next_step = lax_friedrichs_step(rho, t_flow, grid)
    # Torch version
    torch_flow = flows_torch.TriangularFlow(
        rho_c_init=t_flow.rho_c, Q_c_init=t_flow.Q_c, rho_max_init=t_flow.rho_max, rho_free_flow_init=0.2
    )
    rho_torch = torch.tensor(rho, dtype=torch.float32)
    rho_next_torch = lax_friedrichs_step_torch(torch_flow, rho_torch, grid.dt_seconds, grid.dx_meters).detach().numpy()
    assert np.allclose(rho_next_step, rho_next_torch, atol=1e-6)


    # Greenshields Flow
    g_flow = GreenshieldsFlow(rho_max=1.0, v_max=1.0)
    # Numpy version
    rho_next_step = lax_friedrichs_step(rho, g_flow, grid)
    # Torch version
    torch_flow = flows_torch.GreenshieldsFlow(
        rho_max_init=g_flow.rho_max, v_max_init=g_flow.v_max
    )
    rho_next_torch = lax_friedrichs_step_torch(torch_flow, rho_torch, grid.dt_seconds, grid.dx_meters).detach().numpy()
    assert np.allclose(rho_next_step, rho_next_torch, atol=1e-6)


def test_lax_friedrichs_step_equivalence_between_affine_and_step():
    grid = DiscretizationGrid(
        dt_seconds=0.1,
        dx_meters=0.1,
        n_timesteps=5,
        n_cells=5,
    )
    t_flow = TriangularFlow(rho_c=0.5, Q_c=0.5, rho_max=1.0)
    rho = np.linspace(0.1, t_flow.rho_max, grid.n_cells) # shift so we don't end up on rho==rho_c
    rho_next_step = lax_friedrichs_step(rho, t_flow, grid)
    F, B = lax_friedrichs_affine_update(t_flow, rho, grid)
    rho_next_affine = F @ rho + B
    assert np.allclose(rho_next_step, rho_next_affine, atol=1e-6), "Step output mismatch"


def test_lax_friedrichs_step_batch_shape():
    grid = DiscretizationGrid(dx_meters=50, dt_seconds=1, n_timesteps=10, n_cells=20)
    flow = ARZFlow(v_max=30.0, rho_max=0.1, gamma=1.0, tau=5.0)
    U = np.random.rand(4, 2, grid.n_cells) * 0.05
    U_next = lax_friedrichs_step_batch(U, flow, grid)
    assert U_next.shape == (4, 2, grid.n_cells)


def test_arz_gaussian_initial_condition_shape():
    grid = DiscretizationGrid(dx_meters=50, dt_seconds=1, n_timesteps=10, n_cells=20)
    flow = ARZFlow(v_max=30.0, rho_max=0.1, gamma=1.0, tau=5.0)
    U_0 = arz_gaussian_initial_condition(
        grid, flow, rho_background=0.01, rho_peak=0.05, center=500.0, sigma=100.0, n_samples=3
    )
    assert U_0.shape == (3, 2, grid.n_cells)
    assert np.all(U_0[:, 0, :] >= 0)
    assert np.all(U_0[:, 0, :] <= flow.rho_max)


def test_arz_rollout_shape():
    grid = DiscretizationGrid(dx_meters=50, dt_seconds=1, n_timesteps=10, n_cells=20)
    flow = ARZFlow(v_max=30.0, rho_max=0.1, gamma=1.0, tau=5.0)
    U_0 = arz_gaussian_initial_condition(
        grid, flow, rho_background=0.01, rho_peak=0.05, center=500.0, sigma=100.0, n_samples=2
    )
    trajectory = arz_rollout(U_0, flow, grid)
    assert trajectory.shape == (grid.n_timesteps, 2, 2, grid.n_cells)


def test_arz_rollout_preserves_positivity():
    grid = DiscretizationGrid(dx_meters=100, dt_seconds=2, n_timesteps=50, n_cells=30)
    flow = ARZFlow(v_max=25.0, rho_max=0.12, gamma=1.0, tau=10.0)
    U_0 = arz_gaussian_initial_condition(
        grid, flow, rho_background=0.02, rho_peak=0.08, center=1500.0, sigma=200.0, n_samples=1
    )
    trajectory = arz_rollout(U_0, flow, grid)
    assert np.all(trajectory[:, :, 0, :] >= 0), "Density should remain non-negative"
