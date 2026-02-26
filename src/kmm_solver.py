"""
K(m,m) compacton collision solver.

Regularized finite-difference solver with dual spectral filtering
for the K(m,m) equation:
    u_t + (u^m)_x + (u^m)_{xxx} = 0

Numerical method:
  - 4th-order central differences for D1, D3
  - 2nd-order central differences for D2 (dissipation only)
  - Low-pass spectral filter on advection term
  - High-pass spectral filter on artificial dissipation
  - Fixed-step classical RK4 time integration
  - Periodic boundary conditions

Reference: Appendix D of the Lax Centennial paper.
"""

import numpy as np
import time
from scipy.fft import fft, ifft


def build_filters(N, q):
    """Build low-pass and high-pass spectral filters.

    Parameters
    ----------
    N : int
        Grid size.
    q : int
        Filter parameter.  Modes above N/(2q) are suppressed.

    Returns
    -------
    lp, hp : ndarray
        Low-pass and high-pass filter masks in Fourier space.
    """
    n_keep = N // (2 * q)
    lp = np.zeros(N)
    lp[0] = 1.0
    lp[1:n_keep] = 1.0
    lp[N - n_keep + 1:] = 1.0
    hp = 1.0 - lp
    return lp, hp


def compute_cutoff(N, q, L):
    """Physical wavenumber cutoff.

    Returns
    -------
    k_cut_idx : int
        Index of the cutoff mode.
    k_cut_phys : float
        Physical wavenumber cutoff (rad/unit length).
    """
    k_cut_idx = int(np.ceil(N // 2 / q))
    return k_cut_idx, 2 * np.pi * k_cut_idx / L


def solve_kmm(m, u0, x, t_end, q=3, delta=0.5, cfl_safety=0.8,
              progress_interval=30):
    """
    Solve K(m,m) equation with regularized finite differences + RK4.

    RHS(u) = -LP[D1(u^m)] - D3(u^m) + HP[delta * max|u|^{m-1} * dx * D2(u)]

    Parameters
    ----------
    m : float
        Nonlinearity exponent (m > 1).
    u0 : ndarray
        Initial condition on periodic grid.
    x : ndarray
        Spatial grid (uniform, periodic).
    t_end : float
        Final time.
    q : int
        Filter parameter (default 3).
    delta : float
        Dissipation coefficient (default 0.5).
    cfl_safety : float
        CFL safety factor (default 0.8).
    progress_interval : float
        Seconds between progress reports (0 = silent).

    Returns
    -------
    u : ndarray
        Solution at t = t_end.
    wall_time : float
        Wall-clock time in seconds.
    nsteps : int
        Number of RK4 steps taken.
    """
    N = len(x)
    dx = x[1] - x[0]
    lp, hp = build_filters(N, q)

    # Finite difference operators
    def D1(u):  # 4th-order central
        return (-np.roll(u, -2) + 8 * np.roll(u, -1)
                - 8 * np.roll(u, 1) + np.roll(u, 2)) / (12 * dx)

    def D2(u):  # 2nd-order central (HP dissipation only)
        return (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx ** 2)

    def D3(u):  # 4th-order central, 7-point stencil (UNFILTERED)
        return (-np.roll(u, -3) + 8 * np.roll(u, -2) - 13 * np.roll(u, -1)
                + 13 * np.roll(u, 1) - 8 * np.roll(u, 2)
                + np.roll(u, 3)) / (8 * dx ** 3)

    def filt(u, mask):
        return np.real(ifft(fft(u) * mask))

    def rhs(u):
        um = np.sign(u) * np.abs(u) ** m
        umax_m1 = max(np.max(np.abs(u)) ** (m - 1), 1e-10)
        return (-filt(D1(um), lp) - D3(um)
                + filt(delta * umax_m1 * dx * D2(u), hp))

    dt = cfl_safety * 2.83 * dx ** 3 / (4.0 * m)
    u = u0.copy()
    t = 0.0
    nsteps = 0
    t0 = time.time()
    last_report = t0
    total_steps_est = int(np.ceil(t_end / dt))

    while t < t_end - 1e-14:
        dt_use = min(dt, t_end - t)
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt_use * k1)
        k3 = rhs(u + 0.5 * dt_use * k2)
        k4 = rhs(u + dt_use * k3)
        u += (dt_use / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt_use
        nsteps += 1

        if progress_interval > 0:
            now = time.time()
            if now - last_report > progress_interval:
                pct = 100.0 * t / t_end
                elapsed = now - t0
                eta = elapsed / max(pct, 0.1) * (100 - pct)
                print(f"  Step {nsteps:,}/{total_steps_est:,} "
                      f"({pct:.1f}%), elapsed {elapsed:.0f}s, ETA {eta:.0f}s",
                      flush=True)
                last_report = now

    return u, time.time() - t0, nsteps
