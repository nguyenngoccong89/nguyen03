"""
State-space construction and shock discretisation.

Objects built here:
    z_O, z_P       – productivity grids (levels)
    Pi_O, Pi_P     – individual Markov transition matrices
    Pi             – joint transition matrix (Kronecker product, flat indexing)
    h_O, h_P       – default-cost-adjusted productivity  h(z) = z - max(0, d0*z + d1*z^2)
    b_O, b_P       – debt grids (including zero)
    tau            – consumption-tax grid
    iz_O_map, iz_P_map – maps from flat joint z-index to individual indices
"""

import numpy as np
from scipy.stats import norm as _norm


def tauchen(rho, sigma_eps, nz, m):
    """Tauchen (1986) method for AR(1):  ln z' = rho*ln z + eps, eps~N(0, sigma^2).

    Returns
    -------
    z_grid : (nz,) levels  (exp of the log grid)
    Pi     : (nz, nz) transition matrix  Pi[i,j] = Pr(z'=z_j | z=z_i)
    """
    sigma_y = sigma_eps / np.sqrt(1.0 - rho ** 2)
    y_max = m * sigma_y
    y_grid = np.linspace(-y_max, y_max, nz)
    step = y_grid[1] - y_grid[0] if nz > 1 else 1.0

    Pi = np.zeros((nz, nz))
    for i in range(nz):
        mu = rho * y_grid[i]
        for j in range(nz):
            if j == 0:
                Pi[i, j] = _norm.cdf((y_grid[0] + step / 2 - mu) / sigma_eps)
            elif j == nz - 1:
                Pi[i, j] = 1.0 - _norm.cdf((y_grid[-1] - step / 2 - mu) / sigma_eps)
            else:
                Pi[i, j] = (_norm.cdf((y_grid[j] + step / 2 - mu) / sigma_eps)
                            - _norm.cdf((y_grid[j] - step / 2 - mu) / sigma_eps))
    z_grid = np.exp(y_grid)
    return z_grid, Pi


def default_cost(z, d0, d1):
    """h(z) = z - max(0, d0*z + d1*z^2).  Effective productivity under default."""
    return z - np.maximum(0.0, d0 * z + d1 * z ** 2)


def build_grids(p):
    """Construct all grids and precomputed objects.

    Parameters
    ----------
    p : Params

    Returns
    -------
    g : dict  containing all grid arrays and helper mappings
    """
    # Individual productivity grids
    z_O, Pi_O = tauchen(p.rho_z, p.sigma_eps, p.nz_O, p.m_tauchen)
    z_P, Pi_P = tauchen(p.rho_z, p.sigma_eps, p.nz_P, p.m_tauchen)

    # Joint transition matrix  (Kronecker product, flat indexing)
    Pi = np.kron(Pi_O, Pi_P)  # shape (nz, nz)

    # Maps:  flat index iz  →  (iz_O, iz_P)
    iz_O_map = np.repeat(np.arange(p.nz_O), p.nz_P)  # (nz,)
    iz_P_map = np.tile(np.arange(p.nz_P), p.nz_O)     # (nz,)

    # z-levels on the flat grid
    z_O_flat = z_O[iz_O_map]  # (nz,)
    z_P_flat = z_P[iz_P_map]  # (nz,)

    # Default costs
    h_O = default_cost(z_O, p.d0, p.d1)  # (nz_O,)
    h_P = default_cost(z_P, p.d0, p.d1)  # (nz_P,)
    h_O_flat = h_O[iz_O_map]  # (nz,)
    h_P_flat = h_P[iz_P_map]  # (nz,)

    # Debt grids (zero at index 0)
    b_O = np.linspace(0.0, p.bO_max, p.nbO)
    b_P = np.linspace(0.0, p.bP_max, p.nbP)

    # Tax grid
    tau = np.linspace(0.0, p.tau_max, p.ntau)

    return dict(
        z_O=z_O, z_P=z_P,
        Pi_O=Pi_O, Pi_P=Pi_P, Pi=Pi,
        iz_O_map=iz_O_map, iz_P_map=iz_P_map,
        z_O_flat=z_O_flat, z_P_flat=z_P_flat,
        h_O=h_O, h_P=h_P,
        h_O_flat=h_O_flat, h_P_flat=h_P_flat,
        b_O=b_O, b_P=b_P,
        tau=tau,
    )
