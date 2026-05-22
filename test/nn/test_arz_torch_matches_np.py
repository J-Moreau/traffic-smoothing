import numpy as np
import pytest
import torch

from traffic_models.flows import ARZFlow, PowerLawFlux
from traffic_models.nn.arz_high_res import rp1 as rp1_torch
from traffic_models.nn.arz_high_res import weno_reconstruction as weno_torch
from traffic_models.nn.arz_torch import ARZFlow_power_law
from traffic_models.nn.arz_torch import arz_godunov_rollout as arz_rollout_torch
from traffic_models.sim import DiscretizationGrid
from traffic_models.simulate.arz import arz_rollout as arz_rollout_np
from traffic_models.simulate.arz import rp1 as rp1_np
from traffic_models.simulate.arz import weno_reconstruction as weno_np


@pytest.fixture
def flow_params() -> dict[str, float]:
    return {"v_max": 30.0, "rho_max": 0.15, "gamma": 1.0, "tau": 5.0}


@pytest.fixture
def torch_flow(flow_params: dict[str, float]) -> ARZFlow_power_law:
    f = ARZFlow_power_law(
        v_max_init=flow_params["v_max"],
        rho_max_init=flow_params["rho_max"],
        gamma=flow_params["gamma"],
    )
    f.eval()
    return f


@pytest.fixture
def numpy_flow(flow_params: dict[str, float]) -> ARZFlow:
    return ARZFlow(
        flux_function=PowerLawFlux(
            v_max=flow_params["v_max"],
            rho_max=flow_params["rho_max"],
            gamma=flow_params["gamma"],
        ),
        tau=flow_params["tau"],
    )


def _make_smooth_state(n_cells: int, rho_max: float, v_max: float) -> np.ndarray:
    x = np.linspace(0, 1, n_cells)
    rho = 0.02 + 0.08 * np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)
    rho = np.clip(rho, 1e-6, rho_max - 1e-6)
    v = v_max * (1 - rho / rho_max)
    h_val = v_max * (rho / rho_max)
    q = rho * (v + h_val)
    return np.stack([rho, q], axis=0)


class TestRp1:
    def test_output_shapes(self, torch_flow: ARZFlow_power_law) -> None:
        n = 20
        U = _make_smooth_state(n, 0.15, 30.0)
        ql = torch.tensor(U, dtype=torch.float64)
        qr = torch.tensor(U, dtype=torch.float64)
        fwave, s, amdq, apdq = rp1_torch(ql, qr, torch_flow.h, torch_flow.rho_dh)
        assert fwave.shape == (2, 2, n)
        assert s.shape == (2, n)
        assert amdq.shape == (2, n)
        assert apdq.shape == (2, n)

    def test_matches_numpy(self, torch_flow: ARZFlow_power_law, numpy_flow: ARZFlow) -> None:
        n = 5
        U = _make_smooth_state(n, 0.15, 30.0)
        ql_np, qr_np = U.copy(), U.copy()
        ql_t = torch.tensor(U, dtype=torch.float64)
        qr_t = torch.tensor(U, dtype=torch.float64)

        _, _, amdq_np, apdq_np = rp1_np(ql_np, qr_np, numpy_flow.h, numpy_flow.rho_dh)
        _, _, amdq_t, apdq_t = rp1_torch(ql_t, qr_t, torch_flow.h, torch_flow.rho_dh)

        np.testing.assert_allclose(torch_flow.h(ql_t[0]).detach().numpy(), numpy_flow.h(U[0]), atol=1e-6)
        np.testing.assert_allclose(torch_flow.rho_dh(ql_t[0]).detach().numpy(), numpy_flow.rho_dh(U[0]), atol=1e-6)
        np.testing.assert_allclose(amdq_t.detach().numpy(), amdq_np, atol=1e-5)
        np.testing.assert_allclose(apdq_t.detach().numpy(), apdq_np, atol=1e-5)


class TestWenoReconstruction:
    def test_output_shapes(self) -> None:
        n = 20
        U = torch.randn(2, n, dtype=torch.float64)
        Up, Um = weno_torch(U)
        assert Up.shape == (2, n - 4)
        assert Um.shape == (2, n - 4)

    def test_matches_numpy(self) -> None:
        n = 30
        U_np = np.random.RandomState(42).randn(2, n)
        U_t = torch.tensor(U_np, dtype=torch.float64)
        Up_np, Um_np = weno_np(U_np)
        Up_t, Um_t = weno_torch(U_t)
        np.testing.assert_allclose(Up_t.detach().numpy(), Up_np, atol=1e-5)
        np.testing.assert_allclose(Um_t.detach().numpy(), Um_np, atol=1e-5)

    def test_constant_field_is_exact(self) -> None:
        n = 20
        U = torch.ones(2, n, dtype=torch.float64) * 3.0
        Up, Um = weno_torch(U)
        torch.testing.assert_close(Up, torch.full_like(Up, 3.0))
        torch.testing.assert_close(Um, torch.full_like(Um, 3.0))


class TestArzRollout:
    def test_output_shapes(self, torch_flow: ARZFlow_power_law) -> None:
        grid = DiscretizationGrid(dx_meters=100.0, dt_seconds=0.5, n_cells=50, n_timesteps=5)
        U = _make_smooth_state(grid.n_cells, 0.15, 30.0)
        rho_t = torch.tensor(U[0], dtype=torch.float64)
        q_t = torch.tensor(U[1], dtype=torch.float64)
        rho_h, q_h, v_h = arz_rollout_torch(rho_t, q_t, torch_flow, grid)
        assert rho_h.shape == (grid.n_timesteps, grid.n_cells)
        assert q_h.shape == (grid.n_timesteps, grid.n_cells)
        assert v_h.shape == (grid.n_timesteps, grid.n_cells)

    def test_matches_numpy(self, torch_flow: ARZFlow_power_law, numpy_flow: ARZFlow) -> None:
        grid = DiscretizationGrid(dx_meters=100.0, dt_seconds=0.5, n_cells=5, n_timesteps=1)
        U = _make_smooth_state(grid.n_cells, 0.15, 30.0)

        bc = 3
        U_pad = np.pad(U, ((0, 0), (bc, bc)), mode="edge")
        rho_init_np = U_pad[0].copy()
        q_init_np = U_pad[1].copy()
        grid_with_pad = DiscretizationGrid(dx_meters=100.0, dt_seconds=0.5, n_cells=5+2*bc, n_timesteps=1)
        rho_np, q_np, v_np = arz_rollout_np(rho_init_np, q_init_np, numpy_flow, grid_with_pad)
        rho_t = torch.tensor(U[0], dtype=torch.float64)
        q_t = torch.tensor(U[1], dtype=torch.float64)
        rho_th, q_th, v_th = arz_rollout_torch(rho_t, q_t, torch_flow, grid)

        np.testing.assert_allclose(rho_th.detach().numpy(), rho_np[:,bc:-bc], rtol=1e-1)
        np.testing.assert_allclose(q_th.detach().numpy(), q_np[:,bc:-bc], rtol=1e-1)
        np.testing.assert_allclose(v_th.detach().numpy(), v_np[:,bc:-bc], rtol=1e-1)

    def test_density_stays_positive(self, torch_flow: ARZFlow_power_law) -> None:
        grid = DiscretizationGrid(dx_meters=100.0, dt_seconds=0.5, n_cells=50, n_timesteps=20)
        U = _make_smooth_state(grid.n_cells, 0.15, 30.0)
        rho_t = torch.tensor(U[0], dtype=torch.float64)
        q_t = torch.tensor(U[1], dtype=torch.float64)
        rho_h, _, _ = arz_rollout_torch(rho_t, q_t, torch_flow, grid)
        assert (rho_h > 0).all()

def test_arz_godunov_batch_matches_np(numpy_flow: ARZFlow, torch_flow: ARZFlow_power_law) -> None:
    grid = DiscretizationGrid(dx_meters=100.0, dt_seconds=0.5, n_cells=5, n_timesteps=1)
    U = _make_smooth_state(grid.n_cells, 0.15, 30.0)

    bc = 3
    U_pad = np.pad(U, ((0, 0), (bc, bc)), mode="edge")
    grid_with_pad = DiscretizationGrid(dx_meters=100.0, dt_seconds=0.5, n_cells=5 + 2 * bc, n_timesteps=1)
    rho_np, q_np, v_np = arz_rollout_np(U_pad[0].copy(), U_pad[1].copy(), numpy_flow, grid_with_pad)
    rho_t = torch.tensor(U[0], dtype=torch.float64)
    q_t = torch.tensor(U[1], dtype=torch.float64)
    rho_th, q_th, v_th = arz_rollout_torch(rho_t, q_t, torch_flow, grid)

    np.testing.assert_allclose(rho_th.detach().numpy(), rho_np[:, bc:-bc], rtol=1e-1)
    np.testing.assert_allclose(q_th.detach().numpy(), q_np[:, bc:-bc], rtol=1e-1)
    np.testing.assert_allclose(v_th.detach().numpy(), v_np[:, bc:-bc], rtol=1e-1)