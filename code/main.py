#!/usr/bin/env python3
"""
Entry point: solve the two-country sovereign-default model,
simulate, compute moments, and generate all outputs.

Usage:
    python -m code.main
"""

import time
import numpy as np
from code.params import Params
from code.solver import solve_model
from code.simulation import simulate, compute_moments
from code.diagnostics import (
    save_moments_table, plot_time_series, event_study,
    plot_event_study, plot_policy_functions, save_event_study_csv,
    plot_stationary_distributions,
)


def main():
    t0 = time.time()

    # ── 1. Parameters ──
    p = Params()
    print("=" * 60)
    print("Two-Country Sovereign Default Model with Production")
    print("=" * 60)
    print(f"Grid: nz_O={p.nz_O}, nz_P={p.nz_P}, nbO={p.nbO}, nbP={p.nbP}, ntau={p.ntau}")
    print(f"Joint z states: {p.nz}")
    print(f"Convergence: max_iter={p.max_iter}, kappa={p.kappa}")
    print()

    # ── 2. Solve model ──
    print("Solving model...")
    eq = solve_model(p)
    t_solve = time.time() - t0
    print(f"  Solve time: {t_solve:.1f}s")
    print()

    # ── 3. Policy function plots ──
    print("Plotting policy functions...")
    plot_policy_functions(eq)
    print()

    # ── 4. Simulate ──
    print(f"Simulating ({p.sim_N} runs x {p.sim_T} periods)...")
    t1 = time.time()
    sims = simulate(eq, seed=2024)
    t_sim = time.time() - t1
    print(f"  Simulation time: {t_sim:.1f}s")
    print()

    # ── 5. Moments ──
    print("Computing moments...")
    moments = compute_moments(sims, p)
    save_moments_table(moments)
    print()
    print("  Key moments:")
    for k in sorted(moments.keys()):
        print(f"    {k:40s} = {moments[k]:.6f}")
    print()

    # ── 6. Time-series plot ──
    print("Plotting time series...")
    plot_time_series(sims, p, sim_idx=0, T_plot=300)
    print()

    # ── 6b. Stationary distributions ──
    print("Plotting stationary distributions...")
    plot_stationary_distributions(sims, p)
    print()

    # ── 7. Event study around O default ──
    print("Constructing event study around O default events...")
    es = event_study(sims, p)
    plot_event_study(es, p)
    save_event_study_csv(es)
    print()

    t_total = time.time() - t0
    print(f"Total runtime: {t_total:.1f}s")
    print("Done.")


if __name__ == '__main__':
    main()
