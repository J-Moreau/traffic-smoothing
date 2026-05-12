from collections.abc import Callable

import torch


def rp1(
    ql: torch.Tensor,
    qr: torch.Tensor,
    hesitation: Callable[[torch.Tensor], torch.Tensor],
    rho_dhes: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ARZ Riemann solver, translated from sharpclaw code by David Ketcheson"""
    eps = 1e-6
    batch = ql.shape[:-2]
    meqn, n = ql.shape[-2], ql.shape[-1]
    fwave = torch.zeros((*batch, meqn, 2, n), dtype=ql.dtype, device=ql.device)
    s = torch.zeros((*batch, 2, n), dtype=ql.dtype, device=ql.device)
    amdq = torch.zeros((*batch, meqn, n), dtype=ql.dtype, device=ql.device)
    apdq = torch.zeros((*batch, meqn, n), dtype=ql.dtype, device=ql.device)

    j = slice(1, n)
    jm = slice(0, n - 1)
    rho, q = ql[..., 0, j], ql[..., 1, j]
    rhom, qm = qr[..., 0, jm], qr[..., 1, jm]
    h, hm = hesitation(rho), hesitation(rhom)
    rhodh, rhomdhm = rho_dhes(rho), rho_dhes(rhom)
    rho_safe = rho.clamp(min=eps)
    rhom_safe = rhom.clamp(min=eps)

    df1 = (q - rho * h) - (qm - rhom * hm)
    df2 = (q**2 / rho_safe - h * q) - (qm**2 / rhom_safe - hm * qm)

    s1 = qm / rhom_safe - hm - rhomdhm
    s2 = q / rho_safe - h
    s_new = torch.zeros_like(s)
    s_new[..., 0, j] = s1
    s_new[..., 1, j] = s2
    s = s_new

    mask = s1 <= 0

    denL_raw = q / rho_safe - qm / rhom_safe + rhodh
    denL = denL_raw.abs().clamp(min=eps) * torch.sign(denL_raw + eps)
    b1L = ((q / rho_safe + rhodh) * df1 - df2) / denL
    b2L = (-qm / rhom_safe * df1 + df2) / denL

    s1R = q / rho_safe - h - rhodh
    denR = rhodh.clamp(min=eps)
    b1R = ((q / rho_safe + rhodh) * df1 - df2) / denR
    b2R = (-q / rho_safe * df1 + df2) / denR

    fwave_00_j = torch.where(mask, b1L, b1R)
    fwave_10_j = torch.where(mask, b1L * (qm / rhom_safe), b1R * (q / rho_safe))
    fwave_01_j = torch.where(mask, b2L, b2R)
    fwave_11_j = torch.where(
        mask, b2L * (q / rho_safe + rhodh), b2R * (q / rho_safe + rhodh)
    )

    fwave_new = torch.zeros_like(fwave)
    fwave_new[..., 0, 0, j] = fwave_00_j
    fwave_new[..., 1, 0, j] = fwave_10_j
    fwave_new[..., 0, 1, j] = fwave_01_j
    fwave_new[..., 1, 1, j] = fwave_11_j
    fwave = fwave_new

    # Fix s for transonic rarefaction
    s_0_j = torch.where(mask, s1, s1R)
    tr = (~mask) & (s1R <= 0)
    denom_tr_raw = rhom - rho
    denom_tr = denom_tr_raw.abs().clamp(min=eps) * torch.sign(denom_tr_raw + eps)
    s_0_j = torch.where(tr, ((qm - q) + rho * h - rhom * hm) / denom_tr, s_0_j)

    s_final = torch.zeros_like(s)
    s_final[..., 0, j] = s_0_j
    s_final[..., 1, j] = s[..., 1, j]
    s = s_final

    amdq[..., :, j] = (fwave[..., :, :, j] * (s[..., :, j] < 0).unsqueeze(-3)).sum(
        dim=-2
    )
    apdq[..., :, j] = (fwave[..., :, :, j] * (s[..., :, j] >= 0).unsqueeze(-3)).sum(
        dim=-2
    )
    return fwave, s, amdq, apdq


def weno_reconstruction(U: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = U.shape[-1]
    eps = 1e-6
    c1, c2, c3 = 1 / 10, 6 / 10, 3 / 10
    i = torch.arange(2, n - 2, device=U.device)

    # U-[i+1/2]
    q1_0, q1_1, q1_2 = U[..., i - 2], U[..., i - 1], U[..., i]
    q2_0, q2_1, q2_2 = U[..., i - 1], U[..., i], U[..., i + 1]
    q3_0, q3_1, q3_2 = U[..., i], U[..., i + 1], U[..., i + 2]

    u1 = (2 * q1_0 - 7 * q1_1 + 11 * q1_2) / 6
    u2 = (-q2_0 + 5 * q2_1 + 2 * q2_2) / 6
    u3 = (2 * q3_0 + 5 * q3_1 - q3_2) / 6

    b1 = (
        13 / 12 * (q1_0 - 2 * q1_1 + q1_2) ** 2
        + 1 / 4 * (q1_0 - 4 * q1_1 + 3 * q1_2) ** 2
    )
    b2 = 13 / 12 * (q2_0 - 2 * q2_1 + q2_2) ** 2 + 1 / 4 * (q2_0 - q2_2) ** 2
    b3 = (
        13 / 12 * (q3_0 - 2 * q3_1 + q3_2) ** 2
        + 1 / 4 * (3 * q3_0 - 4 * q3_1 + q3_2) ** 2
    )

    alpha1 = c1 / (eps + b1) ** 2
    alpha2 = c2 / (eps + b2) ** 2
    alpha3 = c3 / (eps + b3) ** 2
    alpha_sum = alpha1 + alpha2 + alpha3

    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum
    w3 = alpha3 / alpha_sum
    Um_iphalf = w1 * u1 + w2 * u2 + w3 * u3

    # U+[i-1/2]
    q1r0, q1r1, q1r2 = U[..., i + 2], U[..., i + 1], U[..., i]
    q2r0, q2r1, q2r2 = U[..., i + 1], U[..., i], U[..., i - 1]
    q3r0, q3r1, q3r2 = U[..., i], U[..., i - 1], U[..., i - 2]

    u1r = (2 * q1r0 - 7 * q1r1 + 11 * q1r2) / 6
    u2r = (-q2r0 + 5 * q2r1 + 2 * q2r2) / 6
    u3r = (2 * q3r0 + 5 * q3r1 - q3r2) / 6

    b1r = (
        13 / 12 * (q1r0 - 2 * q1r1 + q1r2) ** 2
        + 1 / 4 * (q1r0 - 4 * q1r1 + 3 * q1r2) ** 2
    )
    b2r = 13 / 12 * (q2r0 - 2 * q2r1 + q2r2) ** 2 + 1 / 4 * (q2r0 - q2r2) ** 2
    b3r = (
        13 / 12 * (q3r0 - 2 * q3r1 + q3r2) ** 2
        + 1 / 4 * (3 * q3r0 - 4 * q3r1 + q3r2) ** 2
    )

    a1r = c1 / (eps + b1r) ** 2
    a2r = c2 / (eps + b2r) ** 2
    a3r = c3 / (eps + b3r) ** 2
    asr = a1r + a2r + a3r

    w1r = a1r / asr
    w2r = a2r / asr
    w3r = a3r / asr
    Up_imhalf = w1r * u1r + w2r * u2r + w3r * u3r

    return Up_imhalf, Um_iphalf

