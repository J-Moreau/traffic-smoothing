import numpy as np
import pytest
import torch

from traffic_models.nn.arz_high_res import rp1, weno_reconstruction
from traffic_models.simulate.arz import rp1 as rp1_np
from traffic_models.simulate.arz import weno_reconstruction as weno_reconstruction_np


def _hesitation(rho: torch.Tensor) -> torch.Tensor:
    return 0.5 * rho


def _rho_dhes(rho: torch.Tensor) -> torch.Tensor:
    return 0.5 * rho


def _hesitation_np(rho: np.ndarray) -> np.ndarray:
    return 0.5 * rho


def _rho_dhes_np(rho: np.ndarray) -> np.ndarray:
    return 0.5 * rho


@pytest.fixture
def sample_ql_qr() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    n = 10
    ql = torch.rand(2, n) + 0.1
    qr = torch.rand(2, n) + 0.1
    return ql, qr


def test_rp1_unbatched_shapes(sample_ql_qr: tuple[torch.Tensor, torch.Tensor]):
    ql, qr = sample_ql_qr
    fwave, s, amdq, apdq = rp1(ql, qr, _hesitation, _rho_dhes)
    n = ql.shape[-1]
    assert fwave.shape == (2, 2, n)
    assert s.shape == (2, n)
    assert amdq.shape == (2, n)
    assert apdq.shape == (2, n)


def test_rp1_batched_shapes(sample_ql_qr: tuple[torch.Tensor, torch.Tensor]):
    ql, qr = sample_ql_qr
    B = 5
    ql_batch = ql.unsqueeze(0).expand(B, -1, -1)
    qr_batch = qr.unsqueeze(0).expand(B, -1, -1)
    fwave, s, amdq, apdq = rp1(ql_batch, qr_batch, _hesitation, _rho_dhes)
    n = ql.shape[-1]
    assert fwave.shape == (B, 2, 2, n)
    assert s.shape == (B, 2, n)
    assert amdq.shape == (B, 2, n)
    assert apdq.shape == (B, 2, n)


def test_rp1_batched_matches_unbatched(sample_ql_qr: tuple[torch.Tensor, torch.Tensor]):
    ql, qr = sample_ql_qr
    fwave_ref, s_ref, amdq_ref, apdq_ref = rp1(ql, qr, _hesitation, _rho_dhes)

    B = 3
    ql_batch = ql.unsqueeze(0).expand(B, -1, -1).clone()
    qr_batch = qr.unsqueeze(0).expand(B, -1, -1).clone()
    fwave_b, s_b, amdq_b, apdq_b = rp1(ql_batch, qr_batch, _hesitation, _rho_dhes)

    for i in range(B):
        torch.testing.assert_close(fwave_b[i], fwave_ref)
        torch.testing.assert_close(s_b[i], s_ref)
        torch.testing.assert_close(amdq_b[i], amdq_ref)
        torch.testing.assert_close(apdq_b[i], apdq_ref)


def test_rp1_flux_conservation(sample_ql_qr: tuple[torch.Tensor, torch.Tensor]):
    ql, qr = sample_ql_qr
    fwave, s, amdq, apdq = rp1(ql, qr, _hesitation, _rho_dhes)
    n = ql.shape[-1]
    j = slice(1, n)
    total = amdq[..., j] + apdq[..., j]
    fwave_sum = fwave[..., :, :, j].sum(dim=-2)
    torch.testing.assert_close(total, fwave_sum)


def test_rp1_torch_matches_numpy(sample_ql_qr: tuple[torch.Tensor, torch.Tensor]):
    ql, qr = sample_ql_qr
    fwave_t, s_t, amdq_t, apdq_t = rp1(ql, qr, _hesitation, _rho_dhes)
    fwave_np, s_np, amdq_np, apdq_np = rp1_np(
        ql.numpy(), qr.numpy(), _hesitation_np, _rho_dhes_np
    )
    torch.testing.assert_close(fwave_t, torch.from_numpy(fwave_np).float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s_t, torch.from_numpy(s_np).float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(amdq_t, torch.from_numpy(amdq_np).float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(apdq_t, torch.from_numpy(apdq_np).float(), atol=1e-5, rtol=1e-5)


def test_weno_reconstruction_torch_matches_numpy():
    torch.manual_seed(0)
    U = torch.rand(2, 20) + 0.1
    up_t, um_t = weno_reconstruction(U)
    up_np, um_np = weno_reconstruction_np(U.numpy())
    torch.testing.assert_close(up_t, torch.from_numpy(up_np).float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(um_t, torch.from_numpy(um_np).float(), atol=1e-5, rtol=1e-5)
