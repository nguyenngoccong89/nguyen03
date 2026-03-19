"""
Main equilibrium solver: iterates the four-case Stackelberg recursion
until convergence of value functions, default probabilities, and
policy-weighted prices.

Steps per iteration (following TeX §Computational Strategy):
  1. Expected continuation objects + price numerators
  2. Choice-conditional bond prices  (Newton on lender Euler)
  3. Tax-optimised repayment returns
  4. Periphery's conditional problem
  5. Outskirt's first-stage Stackelberg problem
  6. Second-stage aggregation  → equilibrium default probabilities
  7. Regime value functions
  8. Tax policy recovery  (for diagnostics)
  9. Policy-weighted prices & lender-marginal-utility update
 10. Damping & convergence check
"""

import numpy as np
from code.params import Params
from code.grids import build_grids
from code.static_block import (
    solve_default_tax, solve_repay_tax_vec, ghh_allocations,
    private_utility, public_utility, _EPS,
)

# ════════════════════════════════════════════════
#  Numerically stable log-sum-exp / softmax
# ════════════════════════════════════════════════

def _logsumexp(v, rho, axis=-1):
    """ρ-scaled log-sum-exp along *axis*.  Returns (lse_value, probabilities)."""
    m = np.max(v, axis=axis, keepdims=True)
    e = np.exp((v - m) / rho)
    s = np.sum(e, axis=axis, keepdims=True)
    lse = m.squeeze(axis) + rho * np.log(s.squeeze(axis))
    prob = e / s
    return lse, prob


def _inclusive_value(v, rho, euler_gamma, axis=-1):
    """Inclusive value  I(v) = γ ρ + m + ρ log Σ exp((v-m)/ρ)."""
    lse, prob = _logsumexp(v, rho, axis)
    return euler_gamma * rho + lse, prob


def _default_prob(V_d, V_r, rho):
    """Smoothed default probability  d = σ((V_d - V_r)/ρ)."""
    diff = (V_d - V_r) / rho
    diff = np.clip(diff, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-diff))


# ════════════════════════════════════════════════
#  Price solver  (vectorised Newton)
# ════════════════════════════════════════════════

def _solve_CL(D, K, sigma_L, CL_min, max_iter, tol):
    """Solve  C_L + K C_L^{σ_L} = D  for C_L by Newton's method.

    D, K : arrays (broadcast-compatible)
    Returns C_L (same shape).
    """
    shape = np.broadcast_shapes(np.shape(D), np.shape(K))
    D = np.broadcast_to(D, shape).copy()
    K = np.broadcast_to(K, shape).copy()
    CL = np.maximum(D / (1.0 + np.abs(K) + 1e-6), CL_min)
    for _ in range(max_iter):
        CL_s = np.maximum(CL, CL_min) ** sigma_L
        f = CL + K * CL_s - D
        fp = 1.0 + sigma_L * K * np.maximum(CL, CL_min) ** (sigma_L - 1.0)
        fp = np.where(np.abs(fp) < 1e-14, 1e-14, fp)
        step = f / fp
        CL = CL - step
        CL = np.maximum(CL, CL_min)
        if np.max(np.abs(step)) < tol:
            break
    return CL


# ════════════════════════════════════════════════
#  Expectation helper
# ════════════════════════════════════════════════

def _Epi(Pi, V):
    """Compute  E[V'](z) = Σ_z' Π(z,z') V(z', ...)  along axis 0.

    Pi : (nz, nz)
    V  : (nz, ...)  —  leading axis indexes current z' (to be summed over)
    Returns array of same shape as V.
    """
    nz = Pi.shape[0]
    flat = V.reshape(nz, -1)
    out = Pi @ flat
    return out.reshape(V.shape)


# ════════════════════════════════════════════════
#  SOLVER
# ════════════════════════════════════════════════

def solve_model(p=None):
    """Solve the two-country Stackelberg sovereign-default model.

    Parameters
    ----------
    p : Params, optional

    Returns
    -------
    eq : dict   —  all equilibrium objects needed for simulation / diagnostics
    """
    if p is None:
        p = Params()

    g = build_grids(p)
    nz  = p.nz
    nbO = p.nbO
    nbP = p.nbP

    Pi     = g['Pi']
    z_O_f  = g['z_O_flat']        # (nz,)
    z_P_f  = g['z_P_flat']        # (nz,)
    h_O_f  = g['h_O_flat']        # (nz,)
    h_P_f  = g['h_P_flat']        # (nz,)
    bO     = g['b_O']             # (nbO,)
    bP     = g['b_P']             # (nbP,)
    tau    = g['tau']             # (ntau,)

    # ── Precompute default returns (depend only on z_i) ──
    RdO, taudO, ldO, cdO, gdO = solve_default_tax(
        g['z_O'], g['h_O'], p.A_O, p, tau)
    RdP, taudP, ldP, cdP, gdP = solve_default_tax(
        g['z_P'], g['h_P'], p.A_P, p, tau)

    iz_O = g['iz_O_map']  # (nz,)
    iz_P = g['iz_P_map']  # (nz,)

    # Broadcast to flat z grid
    RdO_f = RdO[iz_O]     # (nz,)
    RdP_f = RdP[iz_P]     # (nz,)

    # ────────────────────────────────────────────────
    #  Terminal initialisation (closed financial markets)
    # ────────────────────────────────────────────────
    # repayment return at q=0, b'=0: R_r(z_i, b_i, 0, 0)
    def _init_repay(z_flat, b_grid, A):
        """(nz, nb)"""
        zt = z_flat[:, None]   # (nz,1)
        bc = b_grid[None, :]   # (1,nb)
        Rr, _ = solve_repay_tax_vec(zt, bc, 0.0, 0.0, A, p, tau)
        return Rr

    RrO_init = _init_repay(z_O_f, bO, p.A_O)   # (nz, nbO)
    RrP_init = _init_repay(z_P_f, bP, p.A_P)   # (nz, nbP)

    # inclusive value of repay vs default → terminal values
    def _term_val_access(Rr, Rd_f):
        """(nz, nb) from (nz, nb) repay return and (nz,) default return."""
        stack = np.stack([Rr, Rd_f[:, None] * np.ones(Rr.shape[1])[None, :]], axis=-1)
        val, _ = _inclusive_value(stack, p.rho_EV, p.euler_gamma, axis=-1)
        return val

    VO_00 = _term_val_access(RrO_init, RdO_f)[:, :, None] * np.ones(nbP)[None, None, :]
    VP_00 = _term_val_access(RrP_init, RdP_f)[:, None, :] * np.ones(nbO)[None, :, None]

    VO_01 = _term_val_access(RrO_init, RdO_f)  # (nz, nbO)
    VP_10 = _term_val_access(RrP_init, RdP_f)  # (nz, nbP)

    VO_10 = np.tile(RdO_f[:, None], (1, nbP))  # (nz, nbP)
    VP_01 = np.tile(RdP_f[:, None], (1, nbO))  # wait — VP_01 is when P excluded, so depends on (z, bO)
    VP_01 = np.tile(RdP_f[:, None], (1, nbO))  # (nz, nbO)

    VO_11 = RdO_f.copy()      # (nz,)
    VP_11 = RdP_f.copy()      # (nz,)

    # Default probabilities
    dO_00 = _default_prob(RdO_f[:, None, None], RrO_init[:, :, None], p.rho_EV) * np.ones((1, 1, nbP))
    dO_01 = _default_prob(RdO_f[:, None], RrO_init, p.rho_EV)
    dP_00 = _default_prob(RdP_f[:, None, None], RrP_init[:, None, :], p.rho_EV) * np.ones((1, nbO, 1))
    dP_10 = _default_prob(RdP_f[:, None], RrP_init, p.rho_EV)

    # Lender marginal utility
    def _mu(CL, sigma_L):
        return np.maximum(CL, _EPS) ** (-sigma_L)

    mu_L_00 = _mu(p.y_L * np.ones((nz, nbO, nbP)), p.sigma_L)
    mu_L_01 = _mu(p.y_L * np.ones((nz, nbO)), p.sigma_L)
    mu_L_10 = _mu(p.y_L * np.ones((nz, nbP)), p.sigma_L)
    mu_L_11 = _mu(p.y_L * np.ones(nz), p.sigma_L)

    # Policy-weighted prices (initialise to zero)
    qbar_O_00 = np.zeros((nz, nbO, nbP))
    qbar_P_00 = np.zeros((nz, nbO, nbP))
    qbar_O_01 = np.zeros((nz, nbO))
    qbar_P_10 = np.zeros((nz, nbP))

    # ── storage for simulation policy objects ──
    pO_00  = None   # (nz, nbO, nbP, nbO) debt-choice probs O in 00
    pO_01  = None   # (nz, nbO, nbO)
    pP_00  = None   # (nz, nbO, nbP, nbO, nbP) conditional P debt probs in 00
    pP_10  = None   # (nz, nbP, nbP)
    dhatP_00 = None # (nz, nbO, nbP, nbO) P cond. default prob given O repays w/ bO'
    dhatP_10_val = None # (nz, nbP) P default prob when O defaults

    # ────────────────────────────────────────────────
    #  Main iteration
    # ────────────────────────────────────────────────
    for it in range(p.max_iter):

        # ──────── STEP 1: expected continuations + price numerators ────────

        # E_O^00(z, bOp, bPp) = Σ_z' Π V_O^00(z', bOp, bPp)
        EO_00 = _Epi(Pi, VO_00)  # (nz, nbO, nbP)
        EP_00 = _Epi(Pi, VP_00)

        # E_O^01(z, bOp)
        comb_O01 = (1.0 - p.lam_P) * VO_01 + p.lam_P * VO_00[:, :, 0]
        EO_01 = _Epi(Pi, comb_O01)  # (nz, nbO)

        # E_P^01(z, bOp) — P excluded, O active
        comb_P01 = (1.0 - p.lam_P) * VP_01 + p.lam_P * VP_00[:, :, 0]
        EP_01 = _Epi(Pi, comb_P01)  # (nz, nbO)

        # E_O^10(z, bPp)
        comb_O10 = (1.0 - p.lam_O) * VO_10 + p.lam_O * VO_00[:, 0, :]
        EO_10 = _Epi(Pi, comb_O10)  # (nz, nbP)

        # E_P^10(z, bPp) — O excluded, P active
        comb_P10 = (1.0 - p.lam_O) * VP_10 + p.lam_O * VP_00[:, 0, :]
        EP_10 = _Epi(Pi, comb_P10)  # (nz, nbP)

        # E_*^11(z)
        comb_O11 = ((1-p.lam_O)*(1-p.lam_P)*VO_11
                    + p.lam_O*(1-p.lam_P)*VO_01[:, 0]
                    + (1-p.lam_O)*p.lam_P*VO_10[:, 0]
                    + p.lam_O*p.lam_P*VO_00[:, 0, 0])
        EO_11 = _Epi(Pi, comb_O11)  # (nz,)

        comb_P11 = ((1-p.lam_O)*(1-p.lam_P)*VP_11
                    + p.lam_O*(1-p.lam_P)*VP_01[:, 0]
                    + (1-p.lam_O)*p.lam_P*VP_10[:, 0]
                    + p.lam_O*p.lam_P*VP_00[:, 0, 0])
        EP_11 = _Epi(Pi, comb_P11)  # (nz,)

        # Price numerators:
        # N_O^00(z, bOp, bPp) = β_L Σ_z' Π mu_L^00(z',bOp,bPp)(1-d_O^00(z',bOp,bPp))
        MO_00 = mu_L_00 * (1.0 - dO_00)
        NO_00 = p.beta_L * _Epi(Pi, MO_00)

        MP_00 = mu_L_00 * (1.0 - dP_00)
        NP_00 = p.beta_L * _Epi(Pi, MP_00)

        MO_01 = mu_L_01 * (1.0 - dO_01)
        NO_01 = p.beta_L * _Epi(Pi, MO_01)

        MP_10 = mu_L_10 * (1.0 - dP_10)
        NP_10 = p.beta_L * _Epi(Pi, MP_10)

        # ──────── STEP 2: choice-conditional bond prices ────────

        # Regime 00:  q_O^00(z,bO,bP,bOp,bPp),  q_P^00  same shape
        # C_L = y_L + bO + bP - q_O*bOp - q_P*bPp
        # q_O = NO * CL^σL,  q_P = NP * CL^σL
        # → CL + (NO*bOp + NP*bPp) CL^σL = y_L + bO + bP

        # broadcast dimensions:
        #  NO_00 : (nz, nbO_p, nbP_p)  → expand to (nz, 1, 1, nbO, nbP)
        #  bOp   : (nbO,)              → axis 3
        #  bPp   : (nbP,)              → axis 4
        #  bO    : (nbO,)              → axis 1
        #  bP    : (nbP,)              → axis 2

        NO_5 = NO_00[:, None, None, :, :]   # (nz,1,1,nbO,nbP)
        NP_5 = NP_00[:, None, None, :, :]
        bOp_5 = bO[None, None, None, :, None]
        bPp_5 = bP[None, None, None, None, :]
        D_5  = p.y_L + bO[None, :, None, None, None] + bP[None, None, :, None, None]
        K_5  = NO_5 * bOp_5 + NP_5 * bPp_5

        CL_00 = _solve_CL(D_5, K_5, p.sigma_L, p.CL_min, p.price_max_iter, p.price_tol)
        qO_00 = NO_5 * np.maximum(CL_00, _EPS) ** p.sigma_L
        qP_00 = NP_5 * np.maximum(CL_00, _EPS) ** p.sigma_L

        # Regime 01 (only O active): q_O^01(z, bO, bOp)
        NO_3 = NO_01[:, None, :]       # (nz, 1, nbO)
        bOp_3 = bO[None, None, :]      # (1, 1, nbO)
        D_3 = p.y_L + bO[None, :, None]
        K_3 = NO_3 * bOp_3
        CL_01 = _solve_CL(D_3, K_3, p.sigma_L, p.CL_min, p.price_max_iter, p.price_tol)
        qO_01 = NO_3 * np.maximum(CL_01, _EPS) ** p.sigma_L

        # Regime 10 (only P active): q_P^10(z, bP, bPp)
        NP_3 = NP_10[:, None, :]
        bPp_3 = bP[None, None, :]
        D_3p = p.y_L + bP[None, :, None]
        K_3p = NP_3 * bPp_3
        CL_10 = _solve_CL(D_3p, K_3p, p.sigma_L, p.CL_min, p.price_max_iter, p.price_tol)
        qP_10 = NP_3 * np.maximum(CL_10, _EPS) ** p.sigma_L

        # ──────── STEP 3: tax-optimised repayment returns ────────

        # R_O^{r,00}(z, bO, bP, bOp, bPp) — only z_O, bO, bOp, q enter tax problem
        z_O_5 = z_O_f[:, None, None, None, None]
        bO_cur5 = bO[None, :, None, None, None]
        bOp_5t = bO[None, None, None, :, None]
        RO_r00, _ = solve_repay_tax_vec(z_O_5, bO_cur5, bOp_5t, qO_00, p.A_O, p, tau)

        # R_P^{r,00} — only z_P, bP, bPp, q_P enter
        z_P_5 = z_P_f[:, None, None, None, None]
        bP_cur5 = bP[None, None, :, None, None]
        bPp_5t = bP[None, None, None, None, :]
        RP_r00, _ = solve_repay_tax_vec(z_P_5, bP_cur5, bPp_5t, qP_00, p.A_P, p, tau)

        # R_O^{r,01}(z, bO, bOp) — regime 01
        z_O_3 = z_O_f[:, None, None]
        bO_cur3 = bO[None, :, None]
        bOp_3t = bO[None, None, :]
        RO_r01, _ = solve_repay_tax_vec(z_O_3, bO_cur3, bOp_3t, qO_01, p.A_O, p, tau)

        # R_P^{r,10}(z, bP, bPp) — regime 10
        z_P_3 = z_P_f[:, None, None]
        bP_cur3 = bP[None, :, None]
        bPp_3t = bP[None, None, :]
        RP_r10, _ = solve_repay_tax_vec(z_P_3, bP_cur3, bPp_3t, qP_10, p.A_P, p, tau)

        # ──────── STEP 4: Periphery's conditional problem ────────

        # (a) W_P^{00,r}(z,bO,bP,bOp,bPp)
        EP_00_5 = EP_00[:, None, None, :, :]  # (nz,1,1,nbO,nbP)
        WP_00r = RP_r00 + p.beta * EP_00_5    # (nz,nbO,nbP,nbO,nbP)

        # smooth over bPp (axis 4) for each bOp
        # → WhatP_00r(z,bO,bP,bOp)  and  pP_00(bPp | z,bO,bP,bOp)
        WhatP_00r, pP_00_loc = _inclusive_value(
            WP_00r, p.rho_EV, p.euler_gamma, axis=4)  # (nz,nbO,nbP,nbO)

        # (c) W_P^{10,r}(z, bP, bPp)
        EP_10_3 = EP_10[:, None, :]    # (nz,1,nbP)
        WP_10r = RP_r10 + p.beta * EP_10_3   # (nz, nbP, nbP)

        WhatP_10r, pP_10_loc = _inclusive_value(
            WP_10r, p.rho_EV, p.euler_gamma, axis=2)  # (nz, nbP)

        # (e) W_P^{01,d}(z,bO,bP, bOp)   O repays with bOp, P defaults
        EP_01_4 = EP_01[:, None, None, :]    # (nz,1,1,nbO)
        WP_01d = RdP_f[:, None, None, None] + p.beta * EP_01_4  # (nz,1,1,nbO) → broadcast to (nz,nbO,nbP,nbO)
        WP_01d = np.broadcast_to(WP_01d, (nz, nbO, nbP, nbO)).copy()

        # (f) W_P^{11,d}(z,bO,bP)  both default
        WP_11d = RdP_f + p.beta * EP_11  # (nz,)

        # conditional default probabilities (given O's action)
        # hat_d_P^00(z,bO,bP,bOp) when O repays
        dhatP_00_loc = _default_prob(WP_01d, WhatP_00r, p.rho_EV)

        # hat_d_P^10(z,bP) when O defaults
        # WP_11d is (nz,), WhatP_10r is (nz, nbP)
        dhatP_10_loc = _default_prob(
            WP_11d[:, None], WhatP_10r, p.rho_EV)  # (nz, nbP)

        # ──────── STEP 5: Outskirt's first-stage problem ────────

        # (a) W_O^{00,r}(z,bO,bP,bOp,bPp)
        EO_00_5 = EO_00[:, None, None, :, :]
        WO_00r = RO_r00 + p.beta * EO_00_5

        # (b) integrate over P's conditional debt choice
        # pP_00_loc: (nz,nbO,nbP,nbO,nbP) — from _inclusive_value on WP_00r axis=4
        # WO_00r: (nz,nbO,nbP,nbO,nbP)
        # Wtilde_O^{00,r}(z,bO,bP,bOp) = Σ_{bPp} pP_00(bPp|…) W_O^{00,r}(…,bPp)
        WtildeO_00r = np.sum(pP_00_loc * WO_00r, axis=4)  # (nz,nbO,nbP,nbO)

        # (c) W_O^{01,r}(z,bO,bOp) — P defaults, O sole active
        EO_01_3 = EO_01[:, None, :]
        WO_01r = RO_r01 + p.beta * EO_01_3  # (nz, nbO, nbO)

        # (d) W_O^{10,d}(z,bO,bP) — O defaults, P active
        # integrate E_O^10(z, bPp) against p_P^10(bPp|z,bP)
        EO_10_2 = EO_10[:, None, :]     # (nz, 1, nbP)
        # pP_10_loc from axis=2 smooth: shape (nz, nbP, nbP)
        # need Σ_{bPp} pP_10_loc(z,bP,bPp) EO_10(z,bPp)
        EO_10_weighted = np.sum(pP_10_loc * EO_10[:, None, :], axis=2)  # (nz, nbP)
        WO_10d = RdO_f[:, None] + p.beta * EO_10_weighted  # (nz, nbP)

        # (e) W_O^{11,d}(z)  both default
        WO_11d = RdO_f + p.beta * EO_11  # (nz,)

        # smooth over bOp for O's debt-choice
        WhatO_00r, pO_00_loc = _inclusive_value(
            WtildeO_00r, p.rho_EV, p.euler_gamma, axis=3)  # (nz, nbO, nbP)

        WhatO_01r, pO_01_loc = _inclusive_value(
            WO_01r, p.rho_EV, p.euler_gamma, axis=2)       # (nz, nbO)

        # ──────── STEP 6: second-stage Stackelberg aggregation ────────

        # Aggregate P's conditional objects using O's debt probs
        # bar_W_P^{00,r}(z,bO,bP) = Σ_{bOp} pO_00(bOp|s) WhatP_00r(s,bOp)
        barWP_00r = np.sum(pO_00_loc * WhatP_00r, axis=3)  # (nz,nbO,nbP)

        # bar_W_P^{01,d}(z,bO,bP) = Σ_{bOp} pO_00(bOp|s) WP_01d(s,bOp)
        barWP_01d = np.sum(pO_00_loc * WP_01d, axis=3)

        # bar_W_P^{10,r} = WhatP_10r(z,bP)  and  bar_W_P^{11,d} = WP_11d(z)
        barWP_10r = WhatP_10r    # (nz, nbP)
        barWP_11d = WP_11d       # (nz,)

        # P equilibrium default probabilities
        dP_00_new = _default_prob(barWP_01d, barWP_00r, p.rho_EV)  # (nz,nbO,nbP)
        dP_10_new = _default_prob(
            barWP_11d[:, None], barWP_10r, p.rho_EV)               # (nz,nbP)

        # O effective repayment / default values
        barWO_r = (1 - dP_00_new) * WhatO_00r + dP_00_new * WhatO_01r[:, :, None]
        # WhatO_00r: (nz,nbO,nbP), WhatO_01r: (nz,nbO)
        barWO_d_3 = WO_10d  # (nz, nbP) — need to broadcast with bO dim
        barWO_d = ((1 - dP_10_new[:, None, :]) * WO_10d[:, None, :]
                   + dP_10_new[:, None, :] * WO_11d[:, None, None])
        # barWO_d: (nz, nbO, nbP) after broadcast — but WO_10d doesn't depend on bO
        # that's fine: O's default value doesn't depend on bO (debt wiped out)

        dO_00_new = _default_prob(barWO_d, barWO_r, p.rho_EV)  # (nz,nbO,nbP)

        # O in regime 01 (P excluded)
        dO_01_new = _default_prob(
            WO_11d[:, None], WhatO_01r, p.rho_EV)               # (nz,nbO)

        # ──────── STEP 7: regime values ────────

        # O
        VO_00_new = p.euler_gamma * p.rho_EV + np.maximum(barWO_r, barWO_d) \
                    + p.rho_EV * np.log(
                        np.exp((barWO_r - np.maximum(barWO_r, barWO_d)) / p.rho_EV)
                        + np.exp((barWO_d - np.maximum(barWO_r, barWO_d)) / p.rho_EV))

        stack_O01 = np.stack([WhatO_01r, WO_11d[:, None] * np.ones(nbO)[None, :]], axis=-1)
        VO_01_new, _ = _inclusive_value(stack_O01, p.rho_EV, p.euler_gamma, axis=-1)

        VO_10_new = WO_10d.copy()
        VO_11_new = WO_11d.copy()

        # P
        # VP_00 = (1-dO_00) I(barWP_00r, barWP_01d) + dO_00 I(barWP_10r, barWP_11d)
        stack_P00r = np.stack([barWP_00r, barWP_01d], axis=-1)
        I_P00r, _ = _inclusive_value(stack_P00r, p.rho_EV, p.euler_gamma, axis=-1)

        stack_P10r = np.stack([barWP_10r, barWP_11d[:, None] * np.ones(nbP)[None, :]], axis=-1)
        I_P10r, _ = _inclusive_value(stack_P10r, p.rho_EV, p.euler_gamma, axis=-1)

        VP_00_new = (1.0 - dO_00_new) * I_P00r + dO_00_new * I_P10r[:, None, :]
        # I_P10r shape (nz,nbP) → broadcast to (nz, nbO, nbP)

        VP_10_new = I_P10r.copy()

        # V_P^01: P excluded, O has access. From TeX (eq rew_VP01VP11):
        #   V_P^01(s) = bar_W_P^{01,d}(s).
        # In regime 01 the state has bP=0 (index 0).
        # barWP_01d = RdP + β EP_01 already has the right continuation structure.
        VP_01_new = barWP_01d[:, :, 0]  # (nz, nbO)

        VP_11_new = WP_11d.copy()

        # ──────── STEP 9: policy-weighted prices & lender update ────────

        # bar_q_O^00 = Σ Σ pO_00(bOp|s) pP_00(bPp|s,bOp) qO_00(s,bOp,bPp)
        joint_prob = pO_00_loc[..., None] * pP_00_loc  # (nz,nbO,nbP,nbO,nbP)
        qbar_O_00_new = np.sum(joint_prob * qO_00, axis=(3, 4))
        qbar_P_00_new = np.sum(joint_prob * qP_00, axis=(3, 4))

        qbar_O_01_new = np.sum(pO_01_loc * qO_01, axis=2)  # (nz, nbO)
        qbar_P_10_new = np.sum(pP_10_loc * qP_10, axis=2)  # (nz, nbP)

        # policy-weighted lender consumption
        spend_00 = np.sum(joint_prob * (qO_00 * bOp_5 + qP_00 * bPp_5), axis=(3, 4))
        CL_bar_00 = (p.y_L
                     + (1.0 - dO_00_new) * bO[None, :, None]
                     + (1.0 - dP_00_new) * bP[None, None, :]
                     - spend_00)
        CL_bar_00 = np.maximum(CL_bar_00, _EPS)

        spend_01 = np.sum(pO_01_loc * qO_01 * bO[None, None, :], axis=2)
        CL_bar_01 = p.y_L + (1.0 - dO_01_new) * bO[None, :] - spend_01
        CL_bar_01 = np.maximum(CL_bar_01, _EPS)

        spend_10 = np.sum(pP_10_loc * qP_10 * bP[None, None, :], axis=2)
        CL_bar_10 = p.y_L + (1.0 - dP_10_new) * bP[None, :] - spend_10
        CL_bar_10 = np.maximum(CL_bar_10, _EPS)

        mu_L_00_new = _mu(CL_bar_00, p.sigma_L)
        mu_L_01_new = _mu(CL_bar_01, p.sigma_L)
        mu_L_10_new = _mu(CL_bar_10, p.sigma_L)
        mu_L_11_new = p.y_L ** (-p.sigma_L) * np.ones(nz)

        # ──────── STEP 10: damping & convergence ────────

        k = p.kappa
        err_V = max(
            np.max(np.abs(VO_00_new - VO_00)),
            np.max(np.abs(VO_01_new - VO_01)),
            np.max(np.abs(VO_10_new - VO_10)),
            np.max(np.abs(VO_11_new - VO_11)),
            np.max(np.abs(VP_00_new - VP_00)),
            np.max(np.abs(VP_01_new - VP_01)),
            np.max(np.abs(VP_10_new - VP_10)),
            np.max(np.abs(VP_11_new - VP_11)),
        )
        err_d = max(
            np.max(np.abs(dO_00_new - dO_00)),
            np.max(np.abs(dO_01_new - dO_01)),
            np.max(np.abs(dP_00_new - dP_00)),
            np.max(np.abs(dP_10_new - dP_10)),
        )
        err_q = max(
            np.max(np.abs(qbar_O_00_new - qbar_O_00)),
            np.max(np.abs(qbar_P_00_new - qbar_P_00)),
            np.max(np.abs(qbar_O_01_new - qbar_O_01)),
            np.max(np.abs(qbar_P_10_new - qbar_P_10)),
        )

        VO_00 = k * VO_00_new + (1-k) * VO_00
        VO_01 = k * VO_01_new + (1-k) * VO_01
        VO_10 = k * VO_10_new + (1-k) * VO_10
        VO_11 = k * VO_11_new + (1-k) * VO_11
        VP_00 = k * VP_00_new + (1-k) * VP_00
        VP_01 = k * VP_01_new + (1-k) * VP_01
        VP_10 = k * VP_10_new + (1-k) * VP_10
        VP_11 = k * VP_11_new + (1-k) * VP_11

        dO_00 = k * dO_00_new + (1-k) * dO_00
        dO_01 = k * dO_01_new + (1-k) * dO_01
        dP_00 = k * dP_00_new + (1-k) * dP_00
        dP_10 = k * dP_10_new + (1-k) * dP_10

        qbar_O_00 = k * qbar_O_00_new + (1-k) * qbar_O_00
        qbar_P_00 = k * qbar_P_00_new + (1-k) * qbar_P_00
        qbar_O_01 = k * qbar_O_01_new + (1-k) * qbar_O_01
        qbar_P_10 = k * qbar_P_10_new + (1-k) * qbar_P_10

        mu_L_00 = k * mu_L_00_new + (1-k) * mu_L_00
        mu_L_01 = k * mu_L_01_new + (1-k) * mu_L_01
        mu_L_10 = k * mu_L_10_new + (1-k) * mu_L_10
        mu_L_11 = k * mu_L_11_new + (1-k) * mu_L_11

        # store latest simulation objects
        pO_00  = pO_00_loc
        pO_01  = pO_01_loc
        pP_00  = pP_00_loc
        pP_10  = pP_10_loc
        dhatP_00 = dhatP_00_loc
        dhatP_10_val = dhatP_10_loc

        if it % 10 == 0 or (err_V < p.tol_V and err_d < p.tol_d and err_q < p.tol_q):
            print(f"  iter {it:4d}  err_V={err_V:.3e}  err_d={err_d:.3e}  err_q={err_q:.3e}")

        if err_V < p.tol_V and err_d < p.tol_d and err_q < p.tol_q:
            print(f"  *** Converged at iteration {it} ***")
            break
    else:
        print(f"  WARNING: did not converge after {p.max_iter} iterations.")

    # ── pack equilibrium ──
    eq = dict(
        p=p, g=g,
        # value functions
        VO_00=VO_00, VO_01=VO_01, VO_10=VO_10, VO_11=VO_11,
        VP_00=VP_00, VP_01=VP_01, VP_10=VP_10, VP_11=VP_11,
        # default probabilities
        dO_00=dO_00, dO_01=dO_01, dP_00=dP_00, dP_10=dP_10,
        # policy-weighted prices
        qbar_O_00=qbar_O_00, qbar_P_00=qbar_P_00,
        qbar_O_01=qbar_O_01, qbar_P_10=qbar_P_10,
        # lender objects
        mu_L_00=mu_L_00, mu_L_01=mu_L_01, mu_L_10=mu_L_10, mu_L_11=mu_L_11,
        # simulation policy objects
        pO_00=pO_00, pO_01=pO_01,
        pP_00=pP_00, pP_10=pP_10,
        dhatP_00=dhatP_00, dhatP_10=dhatP_10_val,
        # default returns
        RdO_f=RdO_f, RdP_f=RdP_f,
        # default allocations
        RdO=RdO, RdP=RdP, taudO=taudO, taudP=taudP,
        ldO=ldO, cdO=cdO, gdO=gdO,
        ldP=ldP, cdP=cdP, gdP=gdP,
        # branch returns (last iteration)
        WO_10d=WO_10d, WO_11d=WO_11d,
        WP_11d=WP_11d,
        # lender consumption
        CL_bar_00=np.maximum(CL_bar_00, _EPS) if 'CL_bar_00' in dir() else None,
    )
    return eq
