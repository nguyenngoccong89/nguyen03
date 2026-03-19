"""
Diagnostics, event-study construction, and plotting.

Key outputs:
  - Unconditional simulation moments table (CSV)
  - Simulated time-series plots
  - Event-study plots around Outskirt default events
  - Policy-function plots (optional)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import csv


OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TAB_DIR = os.path.join(OUT_DIR, 'tables')

for d in [OUT_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)


def save_moments_table(moments, fname='moments.csv'):
    path = os.path.join(TAB_DIR, fname)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Moment', 'Value'])
        for k, v in sorted(moments.items()):
            w.writerow([k, f'{v:.6f}'])
    print(f"  Moments saved to {path}")


def plot_time_series(sims, p, sim_idx=0, T_plot=200):
    """Plot a sample simulated time series for one simulation run."""
    fig, axes = plt.subplots(6, 2, figsize=(14, 18), sharex=True)
    T_eff = sims['zO'].shape[1]
    t_end = min(T_plot, T_eff)
    t = np.arange(t_end)
    s = {k: v[sim_idx, :t_end] for k, v in sims.items()}

    panels = [
        ('yO', 'Output O', 'yP', 'Output P'),
        ('lO', 'Labor O', 'lP', 'Labor P'),
        ('cO', 'Consumption O', 'cP', 'Consumption P'),
        ('gO', 'Public spending O', 'gP', 'Public spending P'),
        ('tauO', 'Tax rate O', 'tauP', 'Tax rate P'),
        ('spreadO', 'Spread O (bp)', 'spreadP', 'Spread P (bp)'),
    ]

    for row, (kL, labL, kR, labR) in enumerate(panels):
        axes[row, 0].plot(t, s[kL], 'b-', lw=0.7)
        axes[row, 0].set_ylabel(labL)
        axes[row, 1].plot(t, s[kR], 'r-', lw=0.7)
        axes[row, 1].set_ylabel(labR)

        # shade O default events
        dO = s['dO'].astype(bool)
        for ax in [axes[row, 0], axes[row, 1]]:
            for tt in np.where(dO)[0]:
                ax.axvspan(tt - 0.5, tt + 0.5, color='grey', alpha=0.15)

    axes[-1, 0].set_xlabel('Period')
    axes[-1, 1].set_xlabel('Period')
    fig.suptitle('Simulated Time Series', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(FIG_DIR, 'time_series.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Time-series plot saved to {path}")


def build_event_windows(sims, p, event_key='dO', half_window=None):
    """Identify O-default events and build event windows.

    Returns list of (sim_idx, event_time) pairs.  Overlapping events from
    the *same* simulation are handled by requiring a minimum gap of
    2*half_window+1 between consecutive events.
    """
    hw = half_window or p.event_window
    N, T = sims[event_key].shape
    events = []
    for n in range(N):
        d = sims[event_key][n, :]
        ev_times = np.where(d > 0.5)[0]
        if len(ev_times) == 0:
            continue
        selected = [ev_times[0]]
        for tt in ev_times[1:]:
            if tt - selected[-1] > 2 * hw:
                selected.append(tt)
        for tt in selected:
            if tt >= hw and tt + hw < T:
                events.append((n, tt))
    return events


def event_study(sims, p, half_window=None):
    """Construct average dynamics around O default events.

    Returns
    -------
    es : dict  keys → (2*hw+1,) arrays of averaged variables
    """
    hw = half_window or p.event_window
    events = build_event_windows(sims, p, 'dO', hw)
    W = 2 * hw + 1
    n_events = len(events)
    if n_events == 0:
        print("  No O-default events found for event study.")
        return None

    print(f"  Event study: {n_events} O-default events found.")

    var_keys = [
        'yO', 'lO', 'cO', 'gO', 'tauO', 'bO_due', 'spreadO', 'qO', 'dO_prob',
        'yP', 'lP', 'cP', 'gP', 'tauP', 'bP_due', 'spreadP', 'qP', 'dP_prob',
        'CL', 'mu_L', 'dO', 'dP', 'bOp_val', 'bPp_val',
        'repayO', 'repayP', 'purchO', 'purchP',
        'eff_zO', 'eff_zP',
    ]
    accum = {k: np.zeros(W) for k in var_keys}

    for (n, t0) in events:
        for k in var_keys:
            accum[k] += sims[k][n, t0 - hw: t0 + hw + 1]

    es = {k: v / n_events for k, v in accum.items()}
    es['_n_events'] = n_events
    es['_horizon'] = np.arange(-hw, hw + 1)
    return es


def plot_event_study(es, p, half_window=None):
    """Generate event-study plots around O default events."""
    if es is None:
        return
    hw = half_window or p.event_window
    h = es['_horizon']

    # O variables
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    o_vars = [
        ('yO', 'Output'),
        ('lO', 'Labor'),
        ('cO', 'Consumption'),
        ('gO', 'Public spending'),
        ('tauO', 'Tax rate'),
        ('bO_due', 'Debt due'),
        ('spreadO', 'Spread (bp)'),
        ('qO', 'Bond price'),
        ('dO_prob', 'Default prob'),
    ]
    for idx, (k, lab) in enumerate(o_vars):
        ax = axes[idx // 3, idx % 3]
        ax.plot(h, es[k], 'b-o', markersize=3)
        ax.axvline(0, color='grey', ls='--', lw=0.7)
        ax.set_title(f'O: {lab}')
        ax.set_xlabel('Periods from O default')
    fig.suptitle(f'Event Study: O Default (n={es["_n_events"]})', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, 'event_study_O.png'), dpi=150)
    plt.close(fig)

    # P variables
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    p_vars = [
        ('yP', 'Output'),
        ('lP', 'Labor'),
        ('cP', 'Consumption'),
        ('gP', 'Public spending'),
        ('tauP', 'Tax rate'),
        ('bP_due', 'Debt due'),
        ('spreadP', 'Spread (bp)'),
        ('qP', 'Bond price'),
        ('dP_prob', 'Default prob'),
    ]
    for idx, (k, lab) in enumerate(p_vars):
        ax = axes[idx // 3, idx % 3]
        ax.plot(h, es[k], 'r-o', markersize=3)
        ax.axvline(0, color='grey', ls='--', lw=0.7)
        ax.set_title(f'P: {lab}')
        ax.set_xlabel('Periods from O default')
    fig.suptitle(f'Event Study: P around O Default (n={es["_n_events"]})', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, 'event_study_P.png'), dpi=150)
    plt.close(fig)

    # Lender
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    l_vars = [
        ('CL', 'Lender consumption'),
        ('mu_L', 'Lender marginal utility'),
        ('dO', 'O default indicator'),
        ('repayO', 'Repayments from O'),
        ('repayP', 'Repayments from P'),
        ('purchO', 'Purchases of O bonds'),
        ('purchP', 'Purchases of P bonds'),
        ('qO', 'O bond price'),
        ('qP', 'P bond price'),
    ]
    for idx, (k, lab) in enumerate(l_vars):
        ax = axes[idx // 3, idx % 3]
        ax.plot(h, es[k], 'k-o', markersize=3)
        ax.axvline(0, color='grey', ls='--', lw=0.7)
        ax.set_title(lab)
        ax.set_xlabel('Periods from O default')
    fig.suptitle(f'Event Study: Lender around O Default (n={es["_n_events"]})', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, 'event_study_lender.png'), dpi=150)
    plt.close(fig)

    print("  Event-study plots saved.")


def plot_policy_functions(eq):
    """Plot selected policy functions (default probability and bond price)."""
    p = eq['p']
    g = eq['g']
    bO = g['b_O']
    bP = g['b_P']
    nz = p.nz
    iz_mid = nz // 2

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # O default prob in 00
    for ibP in [0, p.nbP // 2, p.nbP - 1]:
        axes[0, 0].plot(bO, eq['dO_00'][iz_mid, :, ibP],
                        label=f'bP={bP[ibP]:.3f}')
    axes[0, 0].set_title('O Default Prob (regime 00, median z)')
    axes[0, 0].set_xlabel('bO')
    axes[0, 0].legend(fontsize=8)

    # P default prob in 00
    for ibO in [0, p.nbO // 2, p.nbO - 1]:
        axes[0, 1].plot(bP, eq['dP_00'][iz_mid, ibO, :],
                        label=f'bO={bO[ibO]:.3f}')
    axes[0, 1].set_title('P Default Prob (regime 00, median z)')
    axes[0, 1].set_xlabel('bP')
    axes[0, 1].legend(fontsize=8)

    # O bond price
    for ibP in [0, p.nbP // 2, p.nbP - 1]:
        axes[1, 0].plot(bO, eq['qbar_O_00'][iz_mid, :, ibP],
                        label=f'bP={bP[ibP]:.3f}')
    axes[1, 0].set_title('O Bond Price (regime 00, median z)')
    axes[1, 0].set_xlabel('bO')
    axes[1, 0].legend(fontsize=8)

    # P bond price
    for ibO in [0, p.nbO // 2, p.nbO - 1]:
        axes[1, 1].plot(bP, eq['qbar_P_00'][iz_mid, ibO, :],
                        label=f'bO={bO[ibO]:.3f}')
    axes[1, 1].set_title('P Bond Price (regime 00, median z)')
    axes[1, 1].set_xlabel('bP')
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'policy_functions.png'), dpi=150)
    plt.close(fig)
    print("  Policy-function plots saved.")


def plot_stationary_distributions(sims, p):
    """Plot stationary distributions of key variables."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    pairs = [
        ('bO_due', 'O Debt Distribution'),
        ('bP_due', 'P Debt Distribution'),
        ('spreadO', 'O Spread Distribution (bp)'),
        ('spreadP', 'P Spread Distribution (bp)'),
        ('CL', 'Lender Consumption Distribution'),
        ('tauO', 'O Tax Rate Distribution'),
    ]
    for idx, (k, lab) in enumerate(pairs):
        ax = axes[idx // 3, idx % 3]
        data = sims[k].ravel()
        data = data[np.isfinite(data)]
        ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
        ax.set_title(lab)
        ax.set_ylabel('Density')

    fig.suptitle('Stationary Distributions', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG_DIR, 'stationary_distributions.png'), dpi=150)
    plt.close(fig)
    print("  Stationary distribution plots saved.")


def save_event_study_csv(es, fname='event_study.csv'):
    """Save event-study averages to CSV."""
    if es is None:
        return
    path = os.path.join(TAB_DIR, fname)
    keys = [k for k in es if not k.startswith('_')]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['horizon'] + keys)
        for i, h in enumerate(es['_horizon']):
            w.writerow([h] + [f'{es[k][i]:.6f}' for k in keys])
    print(f"  Event-study data saved to {path}")
