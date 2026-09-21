"""
emergence_verification.py: independent re-implementation of the Section 6 emergence
experiment in Hyman (2026), "What Remains When Integrability Fails? Mathematical Structure
in the K(m,m) Compacton Equations", DCDS-A (Lax Centennial Volume).

A wide compacton-shaped pulse (A = 1, width factor w = 2.5) evolves under K(m,n) to t = 175;
the script reports every peak at or above the 0.08 floor, to four decimals, beside the printed table.
It runs the four rows compared in ../EMERGENCE_COMPARISON.md: K(2,2), K(3,3), K(4,4), K(3,2).

This is not the production code that produced the table. The paper's stated settings are
followed; settings the paper does not state are fixed as listed in EMERGENCE_COMPARISON.md,
which also records how the output compares with the table.

Run from anywhere:  python papers/hyman2026structure/verification/emergence_verification.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "kmm_compactons"))
from solver import KMNSolver                      # noqa: E402
from stencils import nonlinear_power              # noqa: E402
from filters import apply_lp                      # noqa: E402
from compacton_profiles import compacton_half_support, edge_exponent, compacton_width  # noqa: E402

# Printed table (Section 6), amplitudes above the 0.08 floor; K(3,3) fourth entry carries a dagger.
TABLE = {(2, 2): (1.28, 0.72, 0.34, 0.11),
         (3, 3): (1.20, 0.79, 0.42, 0.10),
         (4, 4): (1.16, 0.84, 0.50),
         (3, 2): (1.33, 0.92, 0.47)}


class KMNmixed(KMNSolver):
    """K(m,n): u_t + (u^m)_x + (u^n)_xxx = 0; the dissipation scaling keeps the advection exponent m."""

    def __init__(self, m, n, **kw):
        super().__init__(m, **kw)
        self.n = float(n)

    def rhs(self, u):
        fx = self._d1(nonlinear_power(u, self.m))
        fxxx = self._d3(nonlinear_power(u, self.n))
        if self.variant == 'A':
            return -apply_lp(fx, self.lp_mask) - fxxx + self._dissipation(u)
        return -apply_lp(fx + fxxx, self.lp_mask) + self._dissipation(u)


def wide_pulse(x, mshape, A=1.0, w=2.5, x0=30.0):
    """K(mshape,mshape) compacton profile of amplitude A, widened by the factor w, centred at x0."""
    al = edge_exponent(mshape)
    ell = compacton_width(mshape) * w
    hw = compacton_half_support(mshape) * w
    xi = x - x0
    u = np.zeros_like(x)
    mk = np.abs(xi) < hw
    u[mk] = A * np.cos(xi[mk] / ell) ** al
    return u


def peaks(u, x, thr=0.08, min_sep=3.0):
    """Local maxima at or above thr, at least min_sep apart, largest first: [(amplitude, x), ...]."""
    idx = [i for i in range(1, len(u) - 1) if u[i] > u[i - 1] and u[i] >= u[i + 1] and u[i] >= thr]
    out = []
    for i in sorted(idx, key=lambda j: -u[j]):
        if all(abs(x[i] - x[j]) > min_sep for _, j in out):
            out.append((u[i], i))
    return sorted([(a, x[i]) for a, i in out], key=lambda t: -t[0])


def run(m, n, delta=0.20, T=175.0, L=250.0, Nx=512, filter_shape='exponential',
        diss_power=4, hp_ramp=True, mshape=None):
    """Evolve the wide pulse under K(m,n) to time T; return (peaks, dt, runtime_s, max u)."""
    s = KMNmixed(m, n, Nx=Nx, L=L, delta=delta, q_lp=2.0, variant='C', spatial='spectral',
                 filter_shape=filter_shape, diss_power=diss_power, hp_ramp=hp_ramp)
    x = s.x
    u0 = wide_pulse(x, mshape=m if mshape is None else mshape)
    dt, _ = s.estimate_dt(u0)
    t0 = time.time()
    _, uh = s.solve(u0, T, dt, save_interval=T)
    u = uh[-1]
    return peaks(u, x), dt, time.time() - t0, u.max()


def main():
    print("Section 6 emergence rows at delta = 0.20, t = 175, N_x = 512 (see EMERGENCE_COMPARISON.md)")
    for (m, n), table in TABLE.items():
        pk, dt, rt, _ = run(m, n)
        amps = ", ".join(f"{a:.4f}" for a, _ in pk)
        print(f"K({m},{n}): N = {len(pk)}  amplitudes = {amps}   table: {', '.join(f'{v:.2f}' for v in table)}"
              f"   (dt = {dt:.4f}, {rt:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
