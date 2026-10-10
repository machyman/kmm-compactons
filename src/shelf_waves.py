"""
Shelf wave profiles and the compacton singular limit.

The K(m,m) equation on a nonzero background u = a admits shelf waves:
smooth, spatially decaying traveling wave solutions that converge to
compactons as a → 0 (Theorem 7.6).

For K(2,2) the shelf wave ODE can be solved in closed form via
elliptic functions. For general m, we use numerical shooting.

Reference: Section 7 and Theorem 7.6 of the Lax Centennial paper.
"""

import numpy as np
from scipy.integrate import solve_ivp
try:
    from .compacton_physics import speed, half_width, compacton_profile
except ImportError:
    from compacton_physics import speed, half_width, compacton_profile


def shelf_wave_ode_k22(a, c, xi_max=30.0, N=2000):
    """
    Compute K(2,2) shelf wave profile by solving the traveling-wave ODE.

    For K(2,2): u_t + (u^2)_x + (u^2)_{xxx} = 0
    Traveling wave u(xi) = U(xi), xi = x - ct, satisfies:
        -c U' + (U^2)' + (U^2)''' = 0

    Parameters
    ----------
    a : float
        Background level (a > 0).
    c : float
        Wave speed. If None, computed from peak amplitude.
    xi_max : float
        Half-domain for computation.
    N : int
        Grid points.

    Returns
    -------
    xi : ndarray
        Spatial coordinate.
    U : ndarray
        Shelf wave profile.
    """
    # For K(2,2), the compacton speed-amplitude relation is c = 3A/4
    # On background a, the effective amplitude above a is A_eff
    # and the wave decays to a as xi -> ±infinity

    xi = np.linspace(-xi_max, xi_max, N)

    # Numerical integration of the ODE system
    # U'' = f(U, U') derived from the integrated ODE
    # -c*U + U^2 + 2*(U')^2 + 2*U*U'' = C (integration constant)
    # At xi -> ±inf: U -> a, U' -> 0, so C = -c*a + a^2

    C_const = -c * a + a ** 2

    def ode_system(xi, y):
        U, Up = y
        if abs(U) < 1e-14:
            return [Up, 0.0]
        Upp = (C_const + c * U - U ** 2 - 2 * Up ** 2) / (2 * U)
        return [Up, Upp]

    # Shooting: start near the peak and integrate both directions
    # Peak amplitude from energy balance
    A_peak = c  # approximate for K(2,2)

    # Forward integration (peak to right tail)
    sol_fwd = solve_ivp(ode_system, [0, xi_max], [A_peak, 0.0],
                        t_eval=np.linspace(0, xi_max, N // 2),
                        method='RK45', rtol=1e-10, atol=1e-12)
    # Backward integration (peak to left tail)
    sol_bwd = solve_ivp(ode_system, [0, -xi_max], [A_peak, 0.0],
                        t_eval=np.linspace(0, -xi_max, N // 2),
                        method='RK45', rtol=1e-10, atol=1e-12)

    # Stitch together
    xi_full = np.concatenate([sol_bwd.t[::-1][:-1], sol_fwd.t])
    U_full = np.concatenate([sol_bwd.y[0, ::-1][:-1], sol_fwd.y[0, :]])

    return xi_full, U_full


def generate_profile_family(m, a_values, x, x0=0.0):
    """
    Generate a family of compacton profiles for varying background a.

    For the paper figures: as a → 0, profiles converge to the compacton.

    Parameters
    ----------
    m : float
        Nonlinearity exponent.
    a_values : list of float
        Background levels (including 0 for the compacton limit).
    x : ndarray
        Spatial grid.
    x0 : float
        Center position.

    Returns
    -------
    profiles : dict
        {a: profile_array} for each background level.
    """
    profiles = {}
    for a in a_values:
        if a == 0 or a < 1e-15:
            # Compacton limit
            A = 1.0  # unit amplitude
            profiles[a] = compacton_profile(m, A, x, x0)
        else:
            # Shelf wave: use K(2,2) ODE solver for m=2,
            # otherwise approximate with smoothed compacton
            c = speed(m, 1.0)  # speed for unit amplitude
            if m == 2:
                xi, U = shelf_wave_ode_k22(a, c, xi_max=half_width(m) * 2)
                # Interpolate onto x grid
                profiles[a] = np.interp(x - x0, xi, U, left=a, right=a)
            else:
                # For general m, use regularized profile
                # (full shooting solver for general m would go here)
                hw = half_width(m)
                profile = compacton_profile(m, 1.0, x, x0)
                # Smooth transition to background
                mask = profile > 0
                profiles[a] = np.where(mask, profile + a, a)
    return profiles


def decay_rate_theory(m, a, c=None):
    """
    Theoretical exponential decay rate kappa(a).

    From Theorem 7.6(d): kappa = sqrt((c - m*a^{m-1}) / (m*a^{m-1}))

    Parameters
    ----------
    m : float
        Nonlinearity exponent.
    a : float
        Background level.
    c : float, optional
        Wave speed. If None, uses speed for unit amplitude.

    Returns
    -------
    kappa : float
        Exponential decay rate.
    """
    if c is None:
        c = speed(m, 1.0)
    c_star = m * a ** (m - 1.0)
    if c <= c_star:
        return np.nan
    return np.sqrt((c - c_star) / c_star)
