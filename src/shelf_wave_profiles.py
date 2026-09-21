#!/usr/bin/env python3
"""
shelf_wave_profiles.py: exact K(2,2) shelf-wave profiles and periods from the quadrature formulas
of Section 7.3 and Appendix J of Hyman (2026), "What Remains When Integrability Fails? Mathematical
Structure in the K(m,m) Compacton Equations", DCDS-A (Peter D. Lax Centennial Memorial Volume).

Solitary shelf waves (a > 0, c > 2a) are known only through the implicit quadrature (eq. quadrature)
    xi(w) = int_w^{w+} 2 s ds / ((s - a) sqrt((w+ - s)(s - w-))),     f = w - a,
with w+- = 2c/3 - a +- sqrt(2c(2c - 3a))/3 (eq. wpm). The compacton limit a = 0 is
w0 = (4c/3) cos^2(xi/4) for |xi| <= 2 pi (eq. compacton-recovery). Cnoidal waves (c < 2a) have period
T(E) = 2 int dv / sqrt(2(E - V(v))) (eq. period-quadrature), V the K(2,2) potential (eq. V-k22).

Parameters are those of the paper's figure data: c = 4 with a = 0, 0.01, 0.10, 0.50, 1.00
(Figure 7), and (a, c) = (2, 3) for the period-energy curve (Supplement Figure S2).

    python src/shelf_wave_profiles.py              compare with the deposited paper/figdata/*.csv
    python src/shelf_wave_profiles.py --write DIR  write the exact values (same grids, same format)

The deposited CSVs came from a separate numerical integration whose settings are not recorded.
The comparison reports how closely they agree with the exact quadrature. Runtime: seconds.
Requirements: NumPy, SciPy, mpmath (installed with SymPy).
"""
import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

C = 4.0                                   # Figure 7 speed
PROFILES = {'profile_compacton.csv': 0.0, 'profile_a0.01.csv': 0.01, 'profile_a0.10.csv': 0.10,
            'profile_a0.50.csv': 0.50, 'profile_a1.00.csv': 1.00}
PERIOD_A, PERIOD_C = 2.0, 3.0              # Supplement Figure S2 (energy measured from the center, V(v0) = 0)


def roots(a, c=C):
    r = np.sqrt(2 * c * (2 * c - 3 * a))
    return 2 * c / 3 - a + r / 3, 2 * c / 3 - a - r / 3


def xi_of_w(w, a, c=C):
    """Implicit quadrature (eq. quadrature), split so both pieces have smooth integrands."""
    wp, wm = roots(a, c)
    sm = a + (wp - a) / 2
    total = 0.0
    if w < sm:                            # s = a + e^u removes the 1/(s - a) growth near the shelf
        g = lambda u: 2 * (a + np.exp(u)) / np.sqrt((wp - a - np.exp(u)) * (a + np.exp(u) - wm))
        total += quad(g, np.log(w - a), np.log(sm - a), epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    lo = max(w, sm)                       # s = w+ - t^2 removes the inverse square root at the crest
    h = lambda t: 4 * (wp - t * t) / ((wp - t * t - a) * np.sqrt(wp - t * t - wm))
    total += quad(h, 0.0, np.sqrt(wp - lo), epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    return total


def profile(a, xis, c=C):
    """f(xi) on a grid; exact compacton for a = 0, inverted quadrature for a > 0."""
    xis = np.asarray(xis, dtype=float)
    if a == 0:
        return np.where(np.abs(xis) <= 2 * np.pi, (4 * c / 3) * np.cos(xis / 4)**2, 0.0)
    wp, _ = roots(a, c)
    A = wp - a
    ylo = np.log(1e-12 * A)
    ximax = xi_of_w(a + np.exp(ylo), a, c)
    cache, out = {}, []
    for x in np.abs(xis):
        key = round(float(x), 12)
        if key not in cache:
            if x == 0:
                cache[key] = A
            elif x >= ximax:
                cache[key] = 0.0          # below 1e-12 of the crest: zero at the printed precision
            else:
                y = brentq(lambda yy: xi_of_w(a + np.exp(yy), a, c) - x, ylo, np.log(A), xtol=1e-14, rtol=1e-14)
                cache[key] = np.exp(y)
        out.append(cache[key])
    return np.array(out)


def period(E, a=PERIOD_A, c=PERIOD_C):
    """T(E) by tanh-sinh quadrature (mpmath), split at the center, with refined turning points."""
    mp.mp.dps = 30
    a, c, E = mp.mpf(a), mp.mpf(c), mp.mpf(E)
    V = lambda v: v**2 / 2 - (2 * c / 3) * v**mp.mpf(1.5) + (c * a - a**2) * v
    v0, vs = a**2, (c - a)**2             # center and saddle
    vl = mp.findroot(lambda v: V(v) - E, (mp.mpf(brentq(lambda t: float(V(t) - E), float(vs) + 1e-14, float(v0))),))
    vh = mp.findroot(lambda v: V(v) - E, (mp.mpf(brentq(lambda t: float(V(t) - E), float(v0), 50.0)),))
    def g(v):                             # tanh-sinh nodes crowd the turning points; beyond working
        d = E - V(v)                      # precision the integrable 1/sqrt tail contributes < 1e-14
        return 1 / mp.sqrt(2 * d) if d > 0 else mp.mpf(0)
    return float(2 * (mp.quad(g, [vl, v0]) + mp.quad(g, [v0, vh])))


def grids():
    lin = np.linspace(-18, 18, 600)
    comp = np.concatenate([np.linspace(-18, -2 * np.pi - 0.01, 50), np.linspace(-2 * np.pi, 2 * np.pi, 500),
                           np.linspace(2 * np.pi + 0.01, 18, 50)])
    return {name: (comp if a == 0 else lin) for name, a in PROFILES.items()}, np.linspace(0.005, 0.495, 100)


def main(argv):
    xgrid, Egrid = grids()
    results = {name: profile(a, xgrid[name]) for name, a in PROFILES.items()}
    results['period_energy.csv'] = np.array([period(E) for E in Egrid])
    if len(argv) > 2 and argv[1] == '--write':
        out = Path(argv[2]); out.mkdir(parents=True, exist_ok=True)
        for name in PROFILES:
            np.savetxt(out / name, np.column_stack([xgrid[name], results[name]]), fmt='%.6f', delimiter=',',
                       header='xi,f', comments='')
        np.savetxt(out / 'period_energy.csv', np.column_stack([Egrid, results['period_energy.csv']]), fmt='%.6f',
                   delimiter=',', header='E,T', comments='')
        print(f"wrote {len(PROFILES) + 1} files to {out}")
        return 0
    figdata = Path(__file__).resolve().parents[1] / 'paper' / 'figdata'
    print(f"Exact quadrature vs deposited figure data in {figdata}")
    print(f"{'file':24s} {'grid match':>11s} {'max |difference|':>17s}  where")
    for name in list(PROFILES) + ['period_energy.csv']:
        d = np.genfromtxt(figdata / name, delimiter=',', names=True)
        x, y = (d['E'], d['T']) if name == 'period_energy.csv' else (d['xi'], d['f'])
        g = Egrid if name == 'period_energy.csv' else xgrid[name]
        gm = np.max(np.abs(np.round(g, 6) - x)) < 1e-9
        err = np.abs(results[name] - y); i = int(np.argmax(err))
        print(f"{name:24s} {str(gm):>11s} {err.max():17.2e}  at {x[i]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
