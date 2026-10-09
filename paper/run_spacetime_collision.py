#!/usr/bin/env python3
"""run_spacetime_collision.py  (hyman2026structure v2_3_0, Session 21)

Space-time history of the K(2,2) collision shown in Figure 3 (fig:collision-k22), for the new
space-time panel requested in review item 13b.

Configuration matches the Figure 3 caption: PSM, N_x = 512, periodic domain [0, 100),
filter variant C with q = 2, biharmonic ramp dissipation delta = 0.15, fixed-step RK4.
Initial data: K(2,2) compactons A1 = 1.0 at x = 25 and A2 = 0.5 at x = 55 (Figure 3a).
Snapshots every 0.25 time units to t = 145.

Output: spacetime_k22.npz (x, t, u) and a validation printout against Figure 3(c):
A1' = 1.000, A2' = 0.498.

Uses the companion-repository package machyman/kmm-compactons, papers/hyman2026structure/src.
Usage: python run_spacetime_collision.py <path-to-src/kmm_compactons> <out-dir>
"""
import sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(src))
from solver import KMNSolver
from compacton_profiles import compacton_profile, compacton_half_support

m, L, A1, A2, X1, X2, T, SAVE = 2, 100.0, 1.0, 0.5, 25.0, 55.0, 145.0, 0.25
s = KMNSolver(m, Nx=512, L=L, delta=0.15, q_lp=2.0, variant='C', spatial='spectral',
              filter_shape='exponential', diss_power=4, hp_ramp=True)
x = s.x
u0 = compacton_profile(x, m, A1, X1, L) + compacton_profile(x, m, A2, X2, L)
t0 = time.time()
dt, info = s.estimate_dt(u0)
print(f"dt = {dt:.6g}  (dt_cfl {info['dt_cfl']:.4g})", flush=True)
th, uh = s.solve(u0, T, dt, save_interval=SAVE)
th, uh = np.array(th), np.array(uh)
print(f"integrated to t = {th[-1]:.3f}: {len(th)} snapshots in {time.time()-t0:.0f} s", flush=True)

def lsfit(u, A_guess, x_guess):
    hs = compacton_half_support(m)
    win = np.abs(((x - x_guess + L/2) % L) - L/2) < hs + 3
    f = lambda p: np.sum((u[win] - compacton_profile(x[win], m, p[0], p[1], L))**2)
    return minimize(f, [A_guess, x_guess], method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-14}).x

u = uh[-1]
idx = [i for i in range(len(u)) if u[i] > u[i-1] and u[i] >= u[(i+1) % len(u)] and u[i] > 0.15*A2]
pk = sorted([(u[i], x[i]) for i in idx], reverse=True)[:2]
fits = sorted([lsfit(u, A, xg) for A, xg in pk], key=lambda p: -p[0])
print(f"t = 145 fit: A1' = {fits[0][0]:.4f} at x = {fits[0][1] % L:.3f};  A2' = {fits[1][0]:.4f} at x = {fits[1][1] % L:.3f}")
print("Figure 3(c) caption: A1' = 1.000, A2' = 0.498")

# free-flight trajectories (no interaction) for the phase-shift measurement
c1 = (m + 1) / (2*m) * A1**(m - 1); c2 = (m + 1) / (2*m) * A2**(m - 1)
d1 = ((fits[0][1] - (X1 + c1*T) + L/2) % L) - L/2
d2 = ((fits[1][1] - (X2 + c2*T) + L/2) % L) - L/2
print(f"speeds c1 = {c1}, c2 = {c2};  phase shifts delta1 = {d1:+.3f}, delta2 = {d2:+.3f}")
np.savez_compressed(out / 'spacetime_k22.npz', x=x, t=th, u=uh, A1=A1, A2=A2, X1=X1, X2=X2,
                    A1p=fits[0][0], A2p=fits[1][0], x1p=fits[0][1] % L, x2p=fits[1][1] % L,
                    delta1=d1, delta2=d2, dt=dt, L=L)
print("wrote", out / 'spacetime_k22.npz', flush=True)
