"""
BBM equation pseudospectral solver.

Solves the Benjamin-Bona-Mahony equation:
    u_t + u_x + u * u_x - u_{xxt} = 0

rewritten as:
    u_t = -(1 - d_xx)^{-1} d_x (u + u^2/2)

using pseudospectral discretization with 2/3 dealiasing and
SciPy's RK45 adaptive integrator.

Reference: Section 5.2, Appendix I and Supplement of the Lax Centennial paper.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.integrate import solve_ivp


def bbm_soliton(A, x, x0=0.0):
    """
    Exact BBM soliton profile.

    u(x) = A * sech^2(kappa * (x - x0))
    where c = 1 + A/3, kappa = sqrt((c-1)/(4c)).

    Parameters
    ----------
    A : float
        Amplitude.
    x : array_like
        Spatial grid.
    x0 : float
        Center position.

    Returns
    -------
    u : ndarray
        Soliton profile.
    """
    c = 1.0 + A / 3.0
    kappa = np.sqrt((c - 1.0) / (4.0 * c))
    return A / np.cosh(kappa * (x - x0)) ** 2


def bbm_speed(A):
    """BBM soliton speed: c = 1 + A/3."""
    return 1.0 + A / 3.0


def solve_bbm(u0, x, t_end, rtol=1e-8, atol=1e-10):
    """
    Solve BBM equation with pseudospectral method + RK45.

    Parameters
    ----------
    u0 : ndarray
        Initial condition on periodic grid.
    x : ndarray
        Spatial grid (uniform, periodic).
    t_end : float
        Final time.
    rtol, atol : float
        Tolerances for RK45 integrator.

    Returns
    -------
    u : ndarray
        Solution at t = t_end.
    info : dict
        Solver diagnostics (nfev, wall_time).
    """
    import time

    N = len(x)
    dx = x[1] - x[0]
    L = N * dx
    k = fftfreq(N, dx) * 2 * np.pi

    # Dealiasing mask (2/3 rule)
    k_max = np.pi / dx
    dealias = np.ones(N)
    dealias[np.abs(k) > 2 * k_max / 3] = 0.0

    # Fourier symbols
    ik = 1j * k
    inv_factor = 1.0 / (1.0 + k ** 2)  # (1 - d_xx)^{-1} in Fourier

    def rhs_flat(t, u):
        u_hat = fft(u)
        # Compute u + u^2/2 with dealiasing
        u_sq = np.real(ifft(dealias * fft(np.real(ifft(u_hat)) ** 2)))
        flux = u + 0.5 * u_sq
        # -(1 - d_xx)^{-1} d_x (flux)
        return np.real(ifft(-inv_factor * ik * fft(flux)))

    t0 = time.time()
    sol = solve_ivp(rhs_flat, [0, t_end], u0, method='RK45',
                    rtol=rtol, atol=atol, dense_output=False,
                    t_eval=[t_end])
    wall_time = time.time() - t0

    return sol.y[:, -1], {
        'nfev': sol.nfev,
        'wall_time': wall_time,
        'success': sol.success,
        'message': sol.message,
    }
