"""Conservation law solvers and utilities for ARZ traffic model."""

import importlib.util
from typing import Callable

import numpy as np

from traffic_models.flows import ARZFlow
from traffic_models.sim import DiscretizationGrid


def lax_friedrichs_flux(F, x, dx, dt):
    # Lax-Friedrichs numerical flux
    F_left = F(x[:, :-1])
    F_right = F(x[:, 1:])
    flux = 0.5 * (F_left + F_right) - 0.5 * (dx/dt) * (x[:, 1:] - x[:, :-1])
    return flux

def local_lax_friedrichs_flux(F, U, h):
    # Local Lax-Friedrichs numerical flux for ARZ
    F_left = F(U[:, :-1])
    F_right = F(U[:, 1:])
    # Compute local wave speeds
    rho_left = U[0, :-1]
    rho_right = U[0, 1:]
    q_left = U[1, :-1]
    q_right = U[1, 1:]
    w_left = q_left / np.maximum(rho_left, 1e-6)
    w_right = q_right / np.maximum(rho_right, 1e-6)
    s_max = np.maximum(np.abs(w_left) + h(rho_left), np.abs(w_right) + h(rho_right))
    flux = 0.5 * (F_left + F_right) - 0.5 * s_max * (U[:, 1:] - U[:, :-1])
    return flux

def HLL_flux(rho_L, w_L, rho_R, w_R, h:Callable, dhes:Callable, ARZ_F:Callable):
    """
    HLL flux for ARZ using eigenvalues:
      lambda1 = u,  lambda2 = u + a, with a = rho P'(rho).
    """
    # Left/right primitive & derived
    u_L = w_L - h(rho_L)
    u_R = w_R - h(rho_R)
    a_L = rho_L*dhes(rho_L)
    a_R = rho_R*dhes(rho_R)

    # Wave speeds
    S_L = np.minimum(u_L, u_R)
    S_R = np.maximum(u_L - a_L, u_R - a_R)

    # Conservative states U = [rho, y]
    U_L = np.stack([rho_L, rho_L * w_L], axis=0)
    U_R = np.stack([rho_R, rho_R * w_R], axis=0)

    F_L = ARZ_F(np.stack([rho_L, w_L], axis=0))
    F_R = ARZ_F(np.stack([rho_R, w_R], axis=0))

    # HLL flux
    denom = (S_R - S_L)
    # Avoid division by zero (degenerate rare case): fallback to average flux
    denom_safe = np.where(denom == 0.0, 1.0, denom)

    FHLL = np.empty_like(F_L)
    # Regions
    mask_L = S_L >= 0.0
    mask_R = S_R <= 0.0
    mask_M = ~(mask_L | mask_R)

    FHLL[:, mask_L] = F_L[:, mask_L]
    FHLL[:, mask_R] = F_R[:, mask_R]
    # Middle (star) state
    num = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L))
    FHLL[:, mask_M] = num[:, mask_M] / denom_safe[mask_M]

    return FHLL


def rp1(ql, qr, hesitation, rho_dhes):
    """Vectorized f-wave Riemann solver for ARZ.
    ql, qr: arrays of shape (2, n) including ghost cells; index j=1..n-1 are interfaces.
    Returns: fwave (2,2,n), s (2,n), amdq (2,n), apdq (2,n).
    
    ! On input, ql contains the state vector at the left edge of each cell
    ! qr contains the state vector at the right edge of each cell

    ! On output, fwave contains the waves as jumps in f,
    ! s the speeds,
    ! 
    ! amdq = A^- Delta q,
    ! apdq = A^+ Delta q,
    ! the decomposition of the flux difference
    ! f(qr(i-1)) - f(ql(i))
    ! into leftgoing and rightgoing parts respectively.

    ! Note that the ith Riemann problem has left state qr(:,i-1)
    !                                  and right state ql(:,i)
    ! From the basic clawpack routines, this routine is called with ql = qr

    translated with copilot from fortran code by David Ketcheson
    """
    eps = 1e-12
    meqn, n = ql.shape
    fwave = np.zeros((meqn, 2, n))
    s = np.zeros((2, n))
    amdq = np.zeros((meqn, n))
    apdq = np.zeros((meqn, n))
    j, jm = slice(1, n), slice(0, n-1)
    rho, q = ql[0, j], ql[1, j]
    rhom, qm = qr[0, jm], qr[1, jm]
    h, hm = hesitation(rho), hesitation(rhom)
    rhodh, rhomdhm = rho_dhes(rho), rho_dhes(rhom)
    rho_safe, rhom_safe = np.maximum(rho, eps), np.maximum(rhom, eps)
    df1 = (q - rho*h) - (qm - rhom*hm)
    df2 = (q**2/rho_safe - h*q) - (qm**2/rhom_safe - hm*qm)
    s1 = qm/rhom_safe - hm - rhomdhm
    s2 = q/rho_safe - h
    s[0, j], s[1, j] = s1, s2
    mask = s1 <= 0
    denL = np.maximum(np.abs(q/rho_safe - qm/rhom_safe + rhodh), eps) * np.sign(q/rho_safe - qm/rhom_safe + rhodh + eps)
    b1L = ((q/rho_safe + rhodh)*df1 - df2)/denL
    b2L = (-qm/rhom_safe*df1 + df2)/denL
    s1R = q/rho_safe - h - rhodh
    denR = np.maximum(rhodh, eps)
    b1R = ((q/rho_safe + rhodh)*df1 - df2)/denR
    b2R = (-q/rho_safe*df1 + df2)/denR
    fwave[0,0,j][mask]  = b1L[mask]
    fwave[1,0,j][mask]  = b1L[mask]*(qm/rhom_safe)[mask]
    fwave[0,1,j][mask]  = b2L[mask]
    fwave[1,1,j][mask]  = b2L[mask]*(q/rho_safe + rhodh)[mask]
    fwave[0,0,j][~mask] = b1R[~mask]
    fwave[1,0,j][~mask] = b1R[~mask]*(q/rho_safe)[~mask]
    fwave[0,1,j][~mask] = b2R[~mask]
    fwave[1,1,j][~mask] = b2R[~mask]*(q/rho_safe + rhodh)[~mask]
    s[0, j][~mask] = s1R[~mask]
    tr = (~mask) & (s1R <= 0)
    denom_tr = np.maximum(np.abs(rhom - rho), eps) * np.sign(rhom - rho + eps)
    s[0, j][tr] = ((qm - q) + rho*h - rhom*hm)[tr] / denom_tr[tr]
    amdq[:, j] = (fwave[:, :, j] * (s[:, j] < 0)[None, :, :]).sum(axis=1)
    apdq[:, j] = (fwave[:, :, j] * (s[:, j] >= 0)[None, :, :]).sum(axis=1)
    return fwave, s, amdq, apdq


def ssp_rk3(U: np.ndarray, dt: float, rhs_func) -> np.ndarray:
    """
    Strong Stability Preserving Runge-Kutta 3rd order time integrator.
    
    Args:
        U: State vector of shape (num_vars, num_cells)
        dt: Time step
        rhs_func: Function that computes dU/dt = rhs_func(U)
    
    Returns:
        U_new: Updated state vector
    """
    # Stage 1: U^(1) = U^n + dt * L(U^n)
    k1 = rhs_func(U)
    U1 = U + dt * k1
    # Stage 2: U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
    k2 = rhs_func(U1)
    U2 = 3/4 * U + 1/4 * (U1 + dt * k2)
    # Stage 3: U^(n+1) = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))
    k3 = rhs_func(U2)
    U_new = 1/3 * U + 2/3 * (U2 + dt * k3)
    return U_new


def weno_reconstruction(U):
    """WENO-5 reconstruction for conservative variables.
    
    Args:
        U: Conservative variables array of shape (num_vars, num_cells)
        
    Returns:
        U_left, U_right: Left and right reconstructed states at interfaces
    """
    num_vars, n = U.shape
    eps = 1e-16  # Small number to avoid division by zero
    # WENO weights
    c1, c2, c3 = 1/10, 6/10, 3/10  # Linear weights
    
    # Create index arrays for vectorized stencil operations
    i = np.arange(2, n-2)  # Interface indices
    
    # U-[i+1/2] RECONSTRUCTION - vectorized stencils
    # Three stencils for all interfaces at once
    
    q1_0 = U[:, i-2]  # Shape: (num_vars, n)
    q1_1 = U[:, i-1]
    q1_2 = U[:, i]
    
    q2_0 = U[:, i-1]
    q2_1 = U[:, i]
    q2_2 = U[:, i+1]
    
    q3_0 = U[:, i]
    q3_1 = U[:, i+1]
    q3_2 = U[:, i+2]
    
    # Candidate reconstructions 
    u1 = (2*q1_0 - 7*q1_1 + 11*q1_2) / 6
    u2 = (-q2_0 + 5*q2_1 + 2*q2_2) / 6
    u3 = (2*q3_0 + 5*q3_1 - q3_2) / 6
    
    # Smoothness indicators 
    b1 = 13/12 * (q1_0 - 2*q1_1 + q1_2)**2 + 1/4 * (q1_0 - 4*q1_1 + 3*q1_2)**2
    b2 = 13/12 * (q2_0 - 2*q2_1 + q2_2)**2 + 1/4 * (q2_0 - q2_2)**2
    b3 = 13/12 * (q3_0 - 2*q3_1 + q3_2)**2 + 1/4 * (3*q3_0 - 4*q3_1 + q3_2)**2
    
    # Weights 
    alpha1 = c1 / (eps + b1)**2
    alpha2 = c2 / (eps + b2)**2
    alpha3 = c3 / (eps + b3)**2
    alpha_sum = alpha1 + alpha2 + alpha3
    
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum
    w3 = alpha3 / alpha_sum
    
    Um_iphalf = w1 * u1 + w2 * u2 + w3 * u3
    
    # U+[i-1/2] RECONSTRUCTION  stencils (reversed order)
    q1_r_0 = U[:, i+2]
    q1_r_1 = U[:, i+1]
    q1_r_2 = U[:, i]
    
    q2_r_0 = U[:, i+1]
    q2_r_1 = U[:, i]
    q2_r_2 = U[:, i-1]
    
    q3_r_0 = U[:, i]
    q3_r_1 = U[:, i-1]
    q3_r_2 = U[:, i-2]
    
    # Candidate reconstructions (flipped) 
    u1_r = (2*q1_r_0 - 7*q1_r_1 + 11*q1_r_2) / 6
    u2_r = (-q2_r_0 + 5*q2_r_1 + 2*q2_r_2) / 6
    u3_r = (2*q3_r_0 + 5*q3_r_1 - q3_r_2) / 6
    
    # Smoothness indicators 
    b1_r = 13/12 * (q1_r_0 - 2*q1_r_1 + q1_r_2)**2 + 1/4 * (q1_r_0 - 4*q1_r_1 + 3*q1_r_2)**2
    b2_r = 13/12 * (q2_r_0 - 2*q2_r_1 + q2_r_2)**2 + 1/4 * (q2_r_0 - q2_r_2)**2
    b3_r = 13/12 * (q3_r_0 - 2*q3_r_1 + q3_r_2)**2 + 1/4 * (3*q3_r_0 - 4*q3_r_1 + q3_r_2)**2
    
    # Weights (reversed order) 
    alpha1_r = c1 / (eps + b1_r)**2
    alpha2_r = c2 / (eps + b2_r)**2
    alpha3_r = c3 / (eps + b3_r)**2
    alpha_sum_r = alpha1_r + alpha2_r + alpha3_r
    
    w1_r = alpha1_r / alpha_sum_r
    w2_r = alpha2_r / alpha_sum_r
    w3_r = alpha3_r / alpha_sum_r
    
    Up_imhalf = w1_r * u1_r + w2_r * u2_r + w3_r * u3_r
    
    return Up_imhalf, Um_iphalf


def arz_godunov_step(
    rho: np.ndarray,
    q: np.ndarray,
    arzflow: ARZFlow,
    grid: DiscretizationGrid,
    U_left: np.ndarray | None = None,
    U_right: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = arzflow.h
    rho_dhes = arzflow.rho_dh
    source_term = arzflow.source_term
    rho_max = arzflow.rho_max
    v_max = arzflow.v_max
    dt = grid.dt_seconds
    dx = grid.dx_meters

    U = np.stack([rho, q], axis=0)
    _Up_imhalf, _Um_iphalf = weno_reconstruction(U)
    Um_iphalf = U.copy()
    Up_imhalf = U.copy()
    Up_imhalf[:, 2:-2] = _Up_imhalf
    Um_iphalf[:, 2:-2] = _Um_iphalf

    left_bc = U_left[:, np.newaxis] if U_left is not None else U[:, 2:3]
    right_bc = U_right[:, np.newaxis] if U_right is not None else U[:, -3:-2]
    Up_imhalf[:, :2] = left_bc
    Up_imhalf[:, -2:] = right_bc
    Um_iphalf[:, :2] = left_bc
    Um_iphalf[:, -2:] = right_bc

    _, _, amdq, apdq = rp1(Up_imhalf, Um_iphalf, h, rho_dhes)
    _, _, amdq2, apdq2 = rp1(Um_iphalf, np.roll(Up_imhalf, -1, axis=1), h, rho_dhes)
    dU = -1 / dx * (apdq + np.roll(amdq, -1, axis=1) + apdq2 + amdq2)
    dU += source_term(U)

    U += dU * dt
    rho_next = U[0]
    q_next = U[1]

    bc = 3
    rho_next[:bc] = left_bc[0] if U_left is not None else rho_next[bc]
    rho_next[-bc:] = right_bc[0] if U_right is not None else rho_next[-bc - 1]
    q_next[:bc] = left_bc[1] if U_left is not None else q_next[bc]
    q_next[-bc:] = right_bc[1] if U_right is not None else q_next[-bc - 1]
    
    eps = 1e-2
    rho_next = np.clip(rho_next, eps*rho_max, rho_max * (1 - eps))
    q_next = np.clip(q_next, eps*rho_max*v_max, 2 * rho_max * v_max * (1- eps))
    v_next = q_next / np.maximum(rho_next, rho_max*eps) - h(rho_next)
    v_next = np.clip(v_next, eps*v_max, v_max*(1 - eps))
    return rho_next, q_next, v_next


def arz_rollout(
    rho: np.ndarray,
    q: np.ndarray,
    arzflow: ARZFlow,
    grid: DiscretizationGrid,
):
    """Custom ARZ Solver with a High Res Godunov scheme and WENO reconstruction."""
    v_eq = arzflow.v_eq
    h = arzflow.h
    rho_dhes = arzflow.rho_dh
    source_term = arzflow.source_term
    rho_max = arzflow.rho_max
    v_max = arzflow.v_max
    Nt = grid.n_timesteps
    Nx = grid.n_cells
    dt = grid.dt_seconds
    dx = grid.dx_meters

    # Compute fluxes
    if q is None:
        v = v_eq(rho)  # Equilibrium velocity
        w = v + h(rho)
        q = rho * w

    rho_hist = np.zeros((Nt, Nx))
    v_hist = np.zeros((Nt, Nx))
    q_hist = np.zeros((Nt, Nx))

    # Time evolution using selected flux scheme
    for n in range(Nt):

        #     
        # consider system U_t + F(U)_x = S(U)
        U = np.stack([rho, q], axis=0)

        def dU_dt(U):
            # Godunov method with f-wave Riemann solver
            _Up_imhalf, _Um_iphalf = weno_reconstruction(U)
            Um_iphalf = U.copy()
            Up_imhalf = U.copy()
            Up_imhalf[:,2:-2] = _Up_imhalf
            Um_iphalf[:,2:-2] = _Um_iphalf
            # periodic boundary conditions
            # Up_imhalf[:,:2], Up_imhalf[:,-2:] = Up_imhalf[:,-2*2:-2], Up_imhalf[:,2:2*2]
            # Um_iphalf[:,:2], Um_iphalf[:,-2:] = Um_iphalf[:,-2*2:-2], Um_iphalf[:,2:2*2]

            # mirror boundary conditions
            Up_imhalf[:,:2], Up_imhalf[:,-2:] = U[:,2], U[:,-3]
            Um_iphalf[:,:2], Um_iphalf[:,-2:] = U[:,2], U[:,-3]

            # Compute Inter-cell fluxes :
            fwave, s, amdq, apdq = rp1(Up_imhalf, Um_iphalf, h, rho_dhes) # expects states u+[i-1/2] and u-[i+1/2]
            # solves riemann problem between u-[i-1/2] and u+[i-1/2] 
            # returns waves at i-1/2
            # a^-delta q = left going fluctuations between i-1, i (idx i-1/2)
            # a^+delta q = right going fluctuations between i-1, i (idx i-1/2)

            # Compute Intra-cell fluxes (high resolution terms):
            fwave, s, amdq2, apdq2 = rp1(
                Um_iphalf,
                np.roll(Up_imhalf,-1,axis=1),
                h, rho_dhes) # expects states u-[i+1/2] and u+[i+1/2]
            # solves riemann problem between u+[i-1/2] and u-[i+1/2]
            # returns waves at i

            # cell i update = a^-delta q_{i+1/2} + a^+delta q_{i-1/2}
            # a^+_{i-1/2} + a^-_{i-1/2} = F(u_i) - F(u_{i-1})
            # and dU/dt = - d(F(U))/dx
            dU = - 1/dx * (apdq + np.roll(amdq,-1,axis=1) + apdq2 + amdq2)
            dU += source_term(U)
            return dU
        
        du_dt = dU_dt(U)
        # U = ssp_rk3(U, dt, dU_dt)
        # or use Euler
        U += du_dt * dt
        rho, q = U[0], U[1]
    
        # Periodic boundary conditions (commented out for open boundary conditions)
        bc = 3  # number of ghost cells
        # rho[:bc], rho[-bc:] = rho[-2*bc:-bc], rho[bc:2*bc]
        # q[:bc], q[-bc:] = q[-2*bc:-bc], q[bc:2*bc]

        # mirror boundary conditions
        rho[:bc], rho[-bc:] = rho[bc], rho[-bc-1]
        q[:bc], q[-bc:] = q[bc], q[-bc-1]
        
        # Clip values to physical bounds
        rho = np.clip(rho, 1e-6, rho_max-1e-6)
        q = np.clip(q, 1e-6, 2 * rho_max * v_max - 1e-6)
        v = q / np.maximum(rho, 1e-6) - h(rho)
        v = np.clip(v, 1e-6, v_max-1e-6)
        rho_hist[n] = rho
        q_hist[n] = q
        v_hist[n] = v

    return rho_hist, q_hist, v_hist


def arz_gaussian_initial_condition(
    grid: DiscretizationGrid,
    flow: ARZFlow,
    rho_background: float,
    rho_peak: float,
    center: float,
    sigma: float,
) -> np.ndarray:
    x = np.linspace(0, grid.n_cells * grid.dx_meters, grid.n_cells)
    rho_0 = rho_background + (rho_peak - rho_background) * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    rho_0 = np.clip(rho_0, 0, flow.rho_max)
    q_0 = (flow.v_eq(rho_0) + flow.h(rho_0)) * rho_0
    U_0 = np.stack([rho_0, q_0], axis=0)
    return U_0
