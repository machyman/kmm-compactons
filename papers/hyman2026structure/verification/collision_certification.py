"""
collision_certification.py — reproduce the K(m,m) two-compacton collision numbers
reported in Hyman (2026, DCDS-A), §5.1, at the production PSM configuration.

Configuration (production PSM): pseudospectral, filter variant C,
q = 2 exponential filter, biharmonic dissipation with smooth ramp, delta = 0.15, N_x = 512,
domain [0, 200], RK4 fixed step (dt from KMNSolver.estimate_dt).

Protocol: larger (faster) compacton at x = 40, smaller at x = 60 (non-overlapping supports);
integrate until the waves have re-separated by more than two support widths; recover each
principal wave by sub-grid least-squares fit of the exact profile (amplitude, position);
debris w = u - (fitted compactons). Report amplitudes, E = (A1'+A2')/(A1+A2), and debris
fractions under both the L2 energy and the Hamiltonian H = ∫u^{m+1}/(m+1).

Run from anywhere:  python papers/hyman2026structure/verification/collision_certification.py
The package in ../src/kmm_compactons/ is put on the import path below. Runtime: under a minute.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'kmm_compactons'))
import compacton_profiles
from solver import KMNSolver
from compacton_profiles import compacton_profile, compacton_half_support

L = 200.0
def make(m):
    return KMNSolver(m, Nx=512, L=L, delta=0.15, q_lp=2.0, variant='C', spatial='spectral',
                     filter_shape='exponential', diss_power=4, hp_ramp=True)

def lsfit(u, x, m, A_guess, x_guess):
    hs = compacton_half_support(m)
    win = np.abs(((x - x_guess + L/2) % L) - L/2) < hs + 3
    f = lambda p: np.sum((u[win] - compacton_profile(x[win], m, p[0], p[1], L))**2)
    return minimize(f, [A_guess, x_guess], method='Nelder-Mead',
                    options={'xatol': 1e-6, 'fatol': 1e-12}).x

def certify(m, A1, A2, T):
    s = make(m); x = s.x
    u0 = compacton_profile(x, m, A1, 40.0, L) + compacton_profile(x, m, A2, 60.0, L)
    dt, _ = s.estimate_dt(u0)
    _, uh = s.solve(u0, T, dt, save_interval=T); u = uh[-1]
    idx = [i for i in range(1, len(u)-1) if u[i] > u[i-1] and u[i] >= u[i+1] and u[i] > 0.15*A2]
    pk = sorted([(u[i], x[i]) for i in idx], reverse=True)[:2]
    fits = sorted([lsfit(u, x, m, A, xg) for A, xg in pk], key=lambda p: -p[0])
    w = u - sum(compacton_profile(x, m, A, x0, L) for A, x0 in fits)
    p = m + 1
    return dict(m=m, A1=A1, A2=A2, T=T, dt=dt,
                A1p=fits[0][0], A2p=fits[1][0], E=(fits[0][0]+fits[1][0])/(A1+A2),
                debris_L2=np.sum(w**2)/np.sum(u**2),
                debris_H=np.sum(np.abs(w)**p)/np.sum(u**p))

if __name__ == '__main__':
    for m, A1, A2, T in [(2, 1.0, 0.8, 200.0), (3, 1.5, 1.0, 100.0)]:
        r = certify(m, A1, A2, T)
        print(f"K({m},{m}) A=({A1},{A2}): A1'={r['A1p']:.4f} A2'={r['A2p']:.4f} "
              f"E={r['E']:.4f}  debris: L2 {100*r['debris_L2']:.4f}%  H {100*r['debris_H']:.5f}%")
