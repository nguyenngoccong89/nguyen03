"""
Monte-Carlo simulation of the solved model.

Given equilibrium policy objects from the solver, simulate T periods:
  - draw productivity shocks (Markov chain on the z-grid)
  - draw readmission shocks
  - in regime 00: O decides default, then debt; P observes and decides
  - in regimes 01/10: sole borrower decides
  - in regime 11: no decisions
  - record all macro variables for O, P, and the lender

Returns a dict of (N, T_eff) arrays where T_eff = T - burn-in.
"""

import numpy as np
from code.static_block import ghh_allocations, _EPS


def _draw_markov(Pi, iz, rng):
    """Draw next state from row Pi[iz, :]."""
    return rng.choice(Pi.shape[1], p=Pi[iz, :])


def _draw_discrete(probs, rng):
    """Draw an index from a probability vector."""
    probs = np.maximum(probs, 0.0)
    s = probs.sum()
    if s < 1e-15:
        return 0
    return rng.choice(len(probs), p=probs / s)


def simulate(eq, seed=42):
    """Run Monte-Carlo simulations.

    Returns
    -------
    sims : dict of arrays, each shape (N, T_eff)
    """
    p = eq['p']
    g = eq['g']
    N = p.sim_N
    T = p.sim_T
    burn = p.sim_burn
    T_eff = T - burn

    Pi   = g['Pi']
    bO   = g['b_O']
    bP   = g['b_P']
    nz   = p.nz
    nbO  = p.nbO
    nbP  = p.nbP
    z_O_f = g['z_O_flat']
    z_P_f = g['z_P_flat']
    h_O_f = g['h_O_flat']
    h_P_f = g['h_P_flat']

    # policy objects
    dO_00  = eq['dO_00']
    dO_01  = eq['dO_01']
    dP_00  = eq['dP_00']
    dP_10  = eq['dP_10']
    pO_00  = eq['pO_00']   # (nz,nbO,nbP,nbO)
    pO_01  = eq['pO_01']   # (nz,nbO,nbO)
    pP_00  = eq['pP_00']   # (nz,nbO,nbP,nbO,nbP)
    pP_10  = eq['pP_10']   # (nz,nbP,nbP)
    dhatP_00 = eq['dhatP_00']  # (nz,nbO,nbP,nbO)
    dhatP_10 = eq['dhatP_10']  # (nz,nbP)

    qbar_O_00 = eq['qbar_O_00']
    qbar_P_00 = eq['qbar_P_00']
    qbar_O_01 = eq['qbar_O_01']
    qbar_P_10 = eq['qbar_P_10']

    taudO = eq['taudO']
    taudP = eq['taudP']

    iz_O_map = g['iz_O_map']
    iz_P_map = g['iz_P_map']

    # allocate storage
    keys = ['iz', 'ibO', 'ibP', 'xO', 'xP',
            'dO', 'dP', 'ibOp', 'ibPp',
            'zO', 'zP', 'bO_due', 'bP_due', 'bOp_val', 'bPp_val',
            'yO', 'yP', 'lO', 'lP', 'cO', 'cP', 'gO', 'gP',
            'tauO', 'tauP', 'qO', 'qP', 'spreadO', 'spreadP',
            'dO_prob', 'dP_prob', 'CL']
    store = {k: np.zeros((N, T)) for k in keys}

    rng = np.random.default_rng(seed)

    for n in range(N):
        # initial state: median z, zero debt, both have access
        iz = nz // 2
        ibO = 0
        ibP = 0
        xO = 0
        xP = 0

        for t in range(T):
            iz_O_val = iz_O_map[iz]
            iz_P_val = iz_P_map[iz]
            zO = z_O_f[iz]
            zP = z_P_f[iz]
            hO = h_O_f[iz]
            hP = h_P_f[iz]

            store['iz'][n, t] = iz
            store['ibO'][n, t] = ibO
            store['ibP'][n, t] = ibP
            store['xO'][n, t] = xO
            store['xP'][n, t] = xP
            store['zO'][n, t] = zO
            store['zP'][n, t] = zP
            store['bO_due'][n, t] = bO[ibO]
            store['bP_due'][n, t] = bP[ibP]

            dO_real = 0
            dP_real = 0
            ibOp = 0
            ibPp = 0
            tauO_val = 0.0
            tauP_val = 0.0

            if xO == 0 and xP == 0:
                # ── regime 00 ──
                d_O_prob = float(dO_00[iz, ibO, ibP])
                dO_real = int(rng.random() < d_O_prob)

                if dO_real == 0:
                    # O repays → draw bOp
                    probs_bOp = pO_00[iz, ibO, ibP, :]
                    ibOp = _draw_discrete(probs_bOp, rng)
                    # P conditional on O repay
                    d_P_cond = float(dhatP_00[iz, ibO, ibP, ibOp])
                    dP_real = int(rng.random() < d_P_cond)
                    if dP_real == 0:
                        probs_bPp = pP_00[iz, ibO, ibP, ibOp, :]
                        ibPp = _draw_discrete(probs_bPp, rng)
                else:
                    ibOp = 0
                    # P conditional on O default
                    d_P_cond = float(dhatP_10[iz, ibP])
                    dP_real = int(rng.random() < d_P_cond)
                    d_O_prob = d_O_prob  # already set
                    if dP_real == 0:
                        probs_bPp = pP_10[iz, ibP, :]
                        ibPp = _draw_discrete(probs_bPp, rng)

                store['dO_prob'][n, t] = dO_00[iz, ibO, ibP]
                store['dP_prob'][n, t] = dP_00[iz, ibO, ibP]
                store['qO'][n, t] = qbar_O_00[iz, ibO, ibP]
                store['qP'][n, t] = qbar_P_00[iz, ibO, ibP]

            elif xO == 0 and xP == 1:
                # ── regime 01 ──
                d_O_prob = float(dO_01[iz, ibO])
                dO_real = int(rng.random() < d_O_prob)
                if dO_real == 0:
                    probs_bOp = pO_01[iz, ibO, :]
                    ibOp = _draw_discrete(probs_bOp, rng)
                dP_real = 0  # P excluded, no decision
                ibPp = 0
                store['dO_prob'][n, t] = dO_01[iz, ibO]
                store['dP_prob'][n, t] = 0.0
                store['qO'][n, t] = qbar_O_01[iz, ibO]
                store['qP'][n, t] = 0.0

            elif xO == 1 and xP == 0:
                # ── regime 10 ──
                dO_real = 0  # O excluded
                d_P_prob = float(dP_10[iz, ibP])
                dP_real = int(rng.random() < d_P_prob)
                if dP_real == 0:
                    probs_bPp = pP_10[iz, ibP, :]
                    ibPp = _draw_discrete(probs_bPp, rng)
                ibOp = 0
                store['dO_prob'][n, t] = 0.0
                store['dP_prob'][n, t] = dP_10[iz, ibP]
                store['qO'][n, t] = 0.0
                store['qP'][n, t] = qbar_P_10[iz, ibP]

            else:
                # ── regime 11 ──
                store['dO_prob'][n, t] = 0.0
                store['dP_prob'][n, t] = 0.0
                store['qO'][n, t] = 0.0
                store['qP'][n, t] = 0.0

            store['dO'][n, t] = dO_real
            store['dP'][n, t] = dP_real
            store['ibOp'][n, t] = ibOp
            store['ibPp'][n, t] = ibPp
            store['bOp_val'][n, t] = bO[ibOp]
            store['bPp_val'][n, t] = bP[ibPp]

            # ── allocations ──
            z_tilde_O = zO if (xO == 0 and dO_real == 0) else hO
            z_tilde_P = zP if (xP == 0 and dP_real == 0) else hP

            # tax: for full accuracy we'd re-solve; use default tax as proxy when defaulting
            if dO_real == 1 or xO == 1:
                tauO_val = float(taudO[iz_O_val])
            else:
                tauO_val = 0.15  # placeholder — will refine below
            if dP_real == 1 or xP == 1:
                tauP_val = float(taudP[iz_P_val])
            else:
                tauP_val = 0.15

            lO, cO, _ = ghh_allocations(tauO_val, z_tilde_O, p.A_O, p.theta, p.nu)
            lP, cP, _ = ghh_allocations(tauP_val, z_tilde_P, p.A_P, p.theta, p.nu)

            yO_out = p.A_O * z_tilde_O * float(lO)
            yP_out = p.A_P * z_tilde_P * float(lP)

            if dO_real == 0 and xO == 0:
                q_O_val = float(store['qO'][n, t])
                gO = tauO_val * float(cO) + q_O_val * bO[ibOp] - bO[ibO]
                # refine: recompute with correct tax
                from code.static_block import solve_repay_tax_vec
                Rr_O, tau_O_opt = solve_repay_tax_vec(
                    z_tilde_O, bO[ibO], bO[ibOp], q_O_val, p.A_O, p, g['tau'])
                tauO_val = float(tau_O_opt)
                lO, cO, _ = ghh_allocations(tauO_val, z_tilde_O, p.A_O, p.theta, p.nu)
                gO = tauO_val * float(cO) + q_O_val * bO[ibOp] - bO[ibO]
                gO = max(gO, 0.0)
            else:
                gO = tauO_val * float(cO)

            if dP_real == 0 and xP == 0:
                q_P_val = float(store['qP'][n, t])
                from code.static_block import solve_repay_tax_vec
                Rr_P, tau_P_opt = solve_repay_tax_vec(
                    z_tilde_P, bP[ibP], bP[ibPp], q_P_val, p.A_P, p, g['tau'])
                tauP_val = float(tau_P_opt)
                lP, cP, _ = ghh_allocations(tauP_val, z_tilde_P, p.A_P, p.theta, p.nu)
                gP = tauP_val * float(cP) + q_P_val * bP[ibPp] - bP[ibP]
                gP = max(gP, 0.0)
            else:
                gP = tauP_val * float(cP)

            store['yO'][n, t] = yO_out
            store['yP'][n, t] = yP_out
            store['lO'][n, t] = float(lO)
            store['lP'][n, t] = float(lP)
            store['cO'][n, t] = float(cO)
            store['cP'][n, t] = float(cP)
            store['gO'][n, t] = float(gO)
            store['gP'][n, t] = float(gP)
            store['tauO'][n, t] = tauO_val
            store['tauP'][n, t] = tauP_val

            # lender consumption
            repayO = (1.0 - dO_real) * bO[ibO] if (xO == 0) else 0.0
            repayP = (1.0 - dP_real) * bP[ibP] if (xP == 0) else 0.0
            purchO = store['qO'][n, t] * bO[ibOp] if (xO == 0 and dO_real == 0) else 0.0
            purchP = store['qP'][n, t] * bP[ibPp] if (xP == 0 and dP_real == 0) else 0.0
            CL_val = p.y_L + repayO + repayP - purchO - purchP
            store['CL'][n, t] = max(CL_val, _EPS)

            # spreads: yield = 1/q - 1; risk-free yield from lender SDF
            # q_rf = β_L * (C_L_next / C_L_now)^{-σ_L} ≈ β_L when consumption stable
            # use model-consistent risk-free price from lender's consumption
            mu_now = max(CL_val, _EPS) ** (-p.sigma_L)
            q_rf = p.beta_L  # will be refined below from avg future mu
            qO_val = store['qO'][n, t]
            qP_val = store['qP'][n, t]
            if qO_val > 1e-6:
                yield_O = 1.0 / qO_val - 1.0
                yield_rf = 1.0 / max(q_rf, 1e-6) - 1.0
                store['spreadO'][n, t] = max(0.0, (yield_O - yield_rf)) * 1e4
            if qP_val > 1e-6:
                yield_P = 1.0 / qP_val - 1.0
                yield_rf = 1.0 / max(q_rf, 1e-6) - 1.0
                store['spreadP'][n, t] = max(0.0, (yield_P - yield_rf)) * 1e4

            # ── transition ──
            iz_new = _draw_markov(Pi, iz, rng)

            if dO_real == 1 or xO == 1:
                xO_new = 1
                ibO_new = 0
                if rng.random() < p.lam_O:
                    xO_new = 0
            else:
                xO_new = 0
                ibO_new = ibOp

            if dP_real == 1 or xP == 1:
                xP_new = 1
                ibP_new = 0
                if rng.random() < p.lam_P:
                    xP_new = 0
            else:
                xP_new = 0
                ibP_new = ibPp

            iz = iz_new
            ibO = ibO_new
            ibP = ibP_new
            xO = xO_new
            xP = xP_new

    # trim burn-in
    sims = {k: v[:, burn:] for k, v in store.items()}
    return sims


def compute_moments(sims, p):
    """Compute unconditional simulation moments."""
    s = sims
    moments = {}

    for prefix, suffix in [('O', 'O'), ('P', 'P')]:
        d_key = f'd{suffix}'
        moments[f'mean_default_rate_{suffix}'] = np.mean(s[d_key])
        moments[f'mean_spread_{suffix}'] = np.mean(s[f'spread{suffix}'])
        moments[f'std_spread_{suffix}'] = np.std(s[f'spread{suffix}'])
        moments[f'mean_debt_gdp_{suffix}'] = np.mean(s[f'b{suffix}_due'] / np.maximum(s[f'y{suffix}'], _EPS))
        moments[f'mean_output_{suffix}'] = np.mean(s[f'y{suffix}'])
        moments[f'mean_consumption_{suffix}'] = np.mean(s[f'c{suffix}'])
        moments[f'mean_labor_{suffix}'] = np.mean(s[f'l{suffix}'])
        moments[f'mean_public_spending_{suffix}'] = np.mean(s[f'g{suffix}'])
        moments[f'mean_tax_{suffix}'] = np.mean(s[f'tau{suffix}'])
        moments[f'mean_default_prob_{suffix}'] = np.mean(s[f'd{suffix}_prob'])
        moments[f'mean_bond_price_{suffix}'] = np.mean(s[f'q{suffix}'])
        moments[f'mean_exclusion_{suffix}'] = np.mean(s[f'x{suffix}'])

    moments['mean_CL'] = np.mean(s['CL'])
    moments['std_CL'] = np.std(s['CL'])
    moments['corr_spreadO_spreadP'] = np.corrcoef(
        s['spreadO'].ravel(), s['spreadP'].ravel())[0, 1]

    return moments
