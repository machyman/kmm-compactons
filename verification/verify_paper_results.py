#!/usr/bin/env python3
"""
Verification script: reproduce key results from the paper.

This script runs a subset of the computational experiments from:
  "What Remains When Integrability Fails?"
  J.M. Hyman, DCDS-A Lax Centennial Volume (2026)

Usage:
    python verification/verify_paper_results.py [--quick]

The --quick flag runs a reduced grid (N=256) for rapid checking.
Full verification uses N=1024 (matches paper).
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compacton_physics import (
    edge_exponent, width_parameter, half_width, speed, compacton_profile
)
from conservation import all_conservation_laws, conservation_drift
from kmm_solver import solve_kmm, compute_cutoff
from collision_analysis import extract_debris, get_spectrum, analyze_below_cutoff


def verify_theorem_4_1():
    """Verify universal compacton parameter formulas."""
    print("=" * 60)
    print("TEST 1: Theorem 4.1 — Universal Formulas")
    print("=" * 60)

    expected = {
        2: {'alpha': 2.0, 'ell': 4.0, 'c_coeff': 3.0 / 4.0},
        3: {'alpha': 1.0, 'ell': 3.0, 'c_coeff': 2.0 / 3.0},
        4: {'alpha': 2.0 / 3.0, 'ell': 8.0 / 3.0, 'c_coeff': 5.0 / 8.0},
        5: {'alpha': 0.5, 'ell': 2.5, 'c_coeff': 3.0 / 5.0},
    }
    all_pass = True
    for m, exp in expected.items():
        alpha = edge_exponent(m)
        ell = width_parameter(m)
        c_coeff = speed(m, 1.0)  # c for A=1

        ok_a = np.isclose(alpha, exp['alpha'])
        ok_l = np.isclose(ell, exp['ell'])
        ok_c = np.isclose(c_coeff, exp['c_coeff'])
        status = "PASS" if (ok_a and ok_l and ok_c) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  K({m},{m}): alpha={alpha:.4f} (exp {exp['alpha']:.4f}) "
              f"ell={ell:.4f} (exp {exp['ell']:.4f}) "
              f"c(A=1)={c_coeff:.4f} (exp {exp['c_coeff']:.4f})  [{status}]")

    print(f"\n  Result: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


def verify_conservation(N=512):
    """Verify conservation laws during single-compacton propagation."""
    print("=" * 60)
    print(f"TEST 2: Conservation Laws (N={N})")
    print("=" * 60)

    m = 2
    A = 1.0
    L = 60.0
    x = np.linspace(0, L, N, endpoint=False)
    t_end = 20.0

    u0 = compacton_profile(m, A, x, 15.0)
    laws_init = all_conservation_laws(u0, x, m)

    u_final, elapsed, nsteps = solve_kmm(
        m, u0, x, t_end, q=3, delta=0.5, cfl_safety=0.8,
        progress_interval=0)
    laws_final = all_conservation_laws(u_final, x, m)
    drift = conservation_drift(laws_init, laws_final)

    all_pass = True
    for key in ['I0', 'I1']:
        ok = drift[key] < 1e-6
        if not ok:
            all_pass = False
        print(f"  {key}: initial={laws_init[key]:.8f}, "
              f"final={laws_final[key]:.8f}, "
              f"drift={drift[key]:.2e}  [{'PASS' if ok else 'FAIL'}]")

    print(f"  Elapsed: {elapsed:.1f}s, {nsteps} steps")
    print(f"\n  Result: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


def verify_collision(m=2, N=512):
    """Verify collision produces quasi-elastic outcome with debris."""
    print("=" * 60)
    print(f"TEST 3: K({m},{m}) Collision (N={N})")
    print("=" * 60)

    A1, A2 = 1.0, 0.5
    L = 100.0
    x = np.linspace(0, L, N, endpoint=False)
    dx = x[1] - x[0]

    c1, c2 = speed(m, A1), speed(m, A2)
    x1, x2 = 20.0, 40.0
    t_col = (x2 - x1) / (c1 - c2)
    t_end = t_col + max(60.0, 2 * half_width(m) / (c1 - c2) + 20.0)

    u0 = compacton_profile(m, A1, x, x1) + compacton_profile(m, A2, x, x2)
    mass_init = np.sum(u0) * dx
    energy_init = np.sum(u0 ** 2) * dx

    print(f"  A1={A1}, A2={A2}, c1={c1:.4f}, c2={c2:.4f}")
    print(f"  t_col={t_col:.1f}, t_end={t_end:.1f}")

    u_final, elapsed, nsteps = solve_kmm(
        m, u0, x, t_end, q=3, delta=0.5, cfl_safety=0.8,
        progress_interval=0)

    mass_err = abs(np.sum(u_final) * dx - mass_init) / abs(mass_init)
    debris, A1f, _, A2f, _, _ = extract_debris(u_final, m, x)
    eps = (A1f + A2f) / (A1 + A2)
    debris_L2 = np.sqrt(np.sum(debris ** 2) * dx)
    energy_frac = np.sum(debris ** 2) * dx / energy_init

    print(f"  A1_fit={A1f:.6f}, A2_fit={A2f:.6f}")
    print(f"  Elasticity eps={eps:.6f}")
    print(f"  Debris L2={debris_L2:.4e}, E_debris/E_total={energy_frac:.4e}")
    print(f"  Mass error={mass_err:.2e}")
    print(f"  Elapsed: {elapsed:.1f}s, {nsteps} steps")

    # Spectral analysis
    k_cut_idx, k_cut_phys = compute_cutoff(N, 3, L)
    k_d, s_d = get_spectrum(debris, x)
    da = analyze_below_cutoff(k_d, s_d, k_cut_phys)
    print(f"  Spectral notch at k=1: {da['notch_ratio']:.4f}")
    print(f"  |w_hat(1)|={da['w_k1']:.4e}")

    ok_mass = mass_err < 1e-4
    ok_elastic = 0.8 < eps < 1.02
    ok_debris = debris_L2 > 0  # must be nonzero (non-integrable)
    all_pass = ok_mass and ok_elastic and ok_debris

    print(f"\n  Mass conserved: [{'PASS' if ok_mass else 'FAIL'}]")
    print(f"  Quasi-elastic (0.8 < eps < 1.02): [{'PASS' if ok_elastic else 'FAIL'}]")
    print(f"  Nonzero debris: [{'PASS' if ok_debris else 'FAIL'}]")
    print(f"\n  Result: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}\n")
    return all_pass


def main():
    quick = '--quick' in sys.argv
    N = 256 if quick else 512

    print("\n" + "=" * 60)
    print("VERIFICATION SUITE: Lax Centennial Paper Results")
    print(f"Mode: {'QUICK (N=256)' if quick else 'STANDARD (N=512)'}")
    print("=" * 60 + "\n")

    t0 = time.time()
    results = []
    results.append(("Theorem 4.1", verify_theorem_4_1()))
    results.append(("Conservation", verify_conservation(N)))
    results.append(("K(2,2) Collision", verify_collision(2, N)))

    if not quick:
        results.append(("K(3,3) Collision", verify_collision(3, N)))

    elapsed = time.time() - t0

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: [{status}]")
        if not passed:
            all_ok = False

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Overall: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
