"""
Parameter specification for the two-country sovereign default model.

Model features: production economy, GHH preferences, distortionary consumption
taxes, public spending, risk-averse lenders, Stackelberg timing (O leads, P follows).

References
----------
- TeX specification (model1903): functional forms, parameters, and computational strategy.
- de Ferra & Mallucci (2025): recursive Stackelberg structure and Appendix C algorithm.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Params:
    # ── Household preferences (GHH) ──
    sigma_H: float = 2.0       # CRRA risk aversion
    theta: float = 1.0         # labour-disutility scale
    nu: float = 2.0            # inverse Frisch elasticity
    omega_g: float = 0.40      # weight on public expenditure
    beta: float = 0.90         # sovereign discount factor

    # ── Productivity process  ln z' = rho * ln z + eps ──
    rho_z: float = 0.90        # persistence
    sigma_eps: float = 0.02    # innovation std dev

    # ── Default cost  h(z) = z - max(0, d0*z + d1*z^2) ──
    d0: float = -0.188
    d1: float = 0.246

    # ── Readmission ──
    lam_O: float = 0.20        # O readmission probability
    lam_P: float = 0.20        # P readmission probability

    # ── Lender ──
    sigma_L: float = 2.5       # lender CRRA risk aversion
    beta_L: float = 0.98       # lender discount factor
    y_L: float = 0.80          # lender endowment

    # ── Country scale ──
    A_O: float = 1.0           # O output scale
    A_P: float = 0.71          # P output scale (de Ferra–Mallucci)

    # ── Extreme-value smoothing ──
    rho_EV: float = 0.50       # EV scale parameter
    euler_gamma: float = 0.5772156649

    # ── Grid sizes ──
    nz_O: int = 5              # O productivity grid points
    nz_P: int = 5              # P productivity grid points
    nbO: int = 8               # O debt grid points
    nbP: int = 8               # P debt grid points
    ntau: int = 50             # tax grid points
    m_tauchen: int = 3         # Tauchen width parameter

    # ── Grid bounds ──
    bO_max: float = 0.20       # maximum O debt
    bP_max: float = 0.20       # maximum P debt
    tau_max: float = 0.50      # maximum tax rate

    # ── Convergence ──
    max_iter: int = 500
    tol_V: float = 5e-4
    tol_d: float = 1e-5
    tol_q: float = 1e-5
    kappa: float = 0.50        # damping  (new = kappa*candidate + (1-kappa)*old)

    # ── Price solver ──
    price_max_iter: int = 60
    price_tol: float = 1e-10
    CL_min: float = 1e-8

    # ── Simulation ──
    sim_T: int = 11000         # total simulation length
    sim_burn: int = 1000       # burn-in periods
    sim_N: int = 50            # number of independent simulations
    event_window: int = 8      # event-study half-window

    @property
    def nz(self):
        """Joint productivity grid size."""
        return self.nz_O * self.nz_P
