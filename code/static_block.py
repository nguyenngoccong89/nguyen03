"""
Static household block and tax optimisation.

Under GHH preferences the intratemporal FOC gives closed-form household
allocations for any (tau, z_tilde, A).  Taxes are then optimised within each
current-period branch (default or repayment) by evaluating the total period
return  u(c,1-l) + v(g)  on the tax grid and picking the maximum.

Key functions
-------------
ghh_allocations   : labour, consumption, composite consumption
private_utility   : u(c_composite)
public_utility    : v(g)
solve_default_tax : optimal tax and period return in default/exclusion
solve_repay_tax   : optimal tax and period return in repayment
"""

import numpy as np

# ── Small floor to keep arguments of power functions positive ──
_EPS = 1e-12


def ghh_allocations(tau, z_tilde, A, theta, nu):
    """
    GHH household allocations.

    l  = (A z_tilde / (theta (1+tau)))^{1/nu}
    c  = A z_tilde l / (1+tau)
    c~ = (nu/(1+nu)) (A z_tilde / (1+tau))^{(1+nu)/nu} / theta^{1/nu}

    All inputs may be broadcastable arrays.  Returns (l, c, c_composite).
    """
    w = A * z_tilde / (1.0 + tau)             # after-tax effective wage
    w_pos = np.maximum(w, _EPS)
    l = (w_pos / theta) ** (1.0 / nu)
    c = w_pos * l
    c_composite = (nu / (1.0 + nu)) * w_pos ** ((1.0 + nu) / nu) / theta ** (1.0 / nu)
    return l, c, np.maximum(c_composite, _EPS)


def private_utility(c_composite, sigma_H):
    """u(c~) = c~^{1-sigma} / (1-sigma).  σ=2 → -1/c~."""
    cc = np.maximum(c_composite, _EPS)
    if sigma_H == 1.0:
        return np.log(cc)
    return cc ** (1.0 - sigma_H) / (1.0 - sigma_H)


def public_utility(g, omega_g, sigma_H):
    """v(g) = omega_g g^{1-sigma} / (1-sigma).  σ=2 → -omega_g/g."""
    gg = np.maximum(g, _EPS)
    if sigma_H == 1.0:
        return omega_g * np.log(gg)
    return omega_g * gg ** (1.0 - sigma_H) / (1.0 - sigma_H)


# ──────────────────────────────────────────────
# Default / exclusion tax problem
# ──────────────────────────────────────────────

def solve_default_tax(z_grid, h_grid, A, p, tau_grid):
    """
    Solve  max_{tau}  u(tau, h(z)) + v(tau c(tau, h(z)))  for each z.

    Parameters
    ----------
    z_grid  : (nz_i,)  productivity grid for country i
    h_grid  : (nz_i,)  default-cost-adjusted productivity
    A       : scalar   country scale
    p       : Params
    tau_grid: (ntau,)

    Returns
    -------
    R_d      : (nz_i,)  optimal period return
    tau_star : (nz_i,)  optimal tax rate
    l_star, c_star, g_star : (nz_i,) allocations at the optimum
    """
    nz = len(z_grid)
    ntau = len(tau_grid)

    zt = h_grid[:, None]        # (nz, 1)
    tt = tau_grid[None, :]      # (1, ntau)

    l, c, cc = ghh_allocations(tt, zt, A, p.theta, p.nu)
    u_val = private_utility(cc, p.sigma_H)

    g = tt * c                  # public spending in default
    v_val = public_utility(g, p.omega_g, p.sigma_H)

    total = u_val + v_val       # (nz, ntau)

    idx = np.argmax(total, axis=1)
    R_d = total[np.arange(nz), idx]
    tau_star = tau_grid[idx]
    l_star = l[np.arange(nz), idx]
    c_star = c[np.arange(nz), idx]
    g_star = g[np.arange(nz), idx]

    return R_d, tau_star, l_star, c_star, g_star


# ──────────────────────────────────────────────
# Repayment tax problem  (vectorised over arbitrary leading dimensions)
# ──────────────────────────────────────────────

def solve_repay_tax_vec(z_tilde, b_current, b_prime, q, A, p, tau_grid):
    """
    Solve  max_{tau}  u(tau, z) + v(tau c(tau,z) + q b' - b)  s.t. g>=0
    for every point in the leading dimensions.

    Parameters
    ----------
    z_tilde   : (*shape)  effective productivity (broadcast-compatible)
    b_current : (*shape)  debt due
    b_prime   : (*shape)  new debt issued
    q         : (*shape)  bond price
    A         : scalar
    p         : Params
    tau_grid  : (ntau,)

    Returns
    -------
    R_r       : (*shape)  optimal period return
    tau_star  : (*shape)  optimal tax
    """
    shape = np.broadcast_shapes(
        np.shape(z_tilde), np.shape(b_current),
        np.shape(b_prime), np.shape(q),
    )
    ntau = len(tau_grid)

    # expand every array to (*shape, 1)  so tau broadcasts along last axis
    zt = np.broadcast_to(z_tilde, shape)[..., None]
    bc = np.broadcast_to(b_current, shape)[..., None]
    bp = np.broadcast_to(b_prime, shape)[..., None]
    qv = np.broadcast_to(q, shape)[..., None]
    tt = tau_grid  # (ntau,) — broadcasts to (..., ntau)

    l, c, cc = ghh_allocations(tt, zt, A, p.theta, p.nu)
    u_val = private_utility(cc, p.sigma_H)

    transfer = qv * bp - bc
    g = tt * c + transfer

    feasible = g > _EPS
    v_val = np.where(feasible,
                     public_utility(np.maximum(g, _EPS), p.omega_g, p.sigma_H),
                     -1e15)
    total = u_val + v_val   # (*shape, ntau)

    idx = np.argmax(total, axis=-1)   # (*shape)
    R_r = np.take_along_axis(total, idx[..., None], axis=-1).squeeze(-1)
    tau_star = tau_grid[idx]

    return R_r, tau_star
