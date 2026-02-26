"""
Core compacton physics for K(m,m) equations.

Implements the universal formulas from Theorem 4.1 of:
  "What Remains When Integrability Fails? Mathematical Structure
   in the K(m,m) Compacton Equations"
  James M. Hyman, DCDS-A Lax Centennial Volume (2026)

The K(m,m) equation:
    u_t + (u^m)_x + (u^m)_{xxx} = 0,  m > 1

admits compacton solutions of the form:
    u(x,t) = A cos^alpha(xi / ell)  for |xi| < pi*ell/2
           = 0                       otherwise
where xi = x - c*t and the parameters satisfy exact algebraic
relations (Theorem 4.1).
"""

import numpy as np


def edge_exponent(m):
    """Edge exponent alpha = 2/(m-1).  Theorem 4.1."""
    return 2.0 / (m - 1.0)


def width_parameter(m):
    """Width parameter ell = 2m/(m-1).  Theorem 4.1."""
    return 2.0 * m / (m - 1.0)


def half_width(m):
    """Half-width of compact support: pi*ell/2."""
    return np.pi * width_parameter(m) / 2.0


def speed(m, A):
    """Speed-amplitude relation c = (m+1)/(2m) * A^{m-1}.  Theorem 4.1."""
    return (m + 1.0) / (2.0 * m) * A ** (m - 1.0)


def amplitude_from_speed(m, c):
    """Inverse: A = (2mc/(m+1))^{1/(m-1)}."""
    return (2.0 * m * c / (m + 1.0)) ** (1.0 / (m - 1.0))


def mass_coefficient(m):
    """C_m such that I_0 = C_m * A.  Eq. (4.7)."""
    from scipy.special import beta as beta_func
    alpha = edge_exponent(m)
    ell = width_parameter(m)
    return ell * beta_func(0.5, alpha / 2.0 + 0.5)


def energy_coefficient(m):
    """D_m such that I_1 = D_m * A^{m+1}.  Eq. (4.8)."""
    from scipy.special import beta as beta_func
    alpha = edge_exponent(m)
    ell = width_parameter(m)
    return ell * beta_func(0.5, (m + 1) * alpha / 2.0 + 0.5)


def compacton_profile(m, A, x, x0=0.0):
    """
    Evaluate the K(m,m) compacton profile.

    Parameters
    ----------
    m : float
        Nonlinearity exponent (m > 1).
    A : float
        Amplitude.
    x : array_like
        Spatial grid (1D).
    x0 : float
        Center position.

    Returns
    -------
    u : ndarray
        Compacton profile, zero outside compact support.
    """
    alpha = edge_exponent(m)
    ell = width_parameter(m)
    hw = np.pi * ell / 2.0
    xi = np.asarray(x, dtype=np.float64) - x0
    u = np.zeros_like(xi)
    inside = np.abs(xi) < hw
    u[inside] = A * np.cos(xi[inside] / ell) ** alpha
    return u


def shelf_wave_decay_rate(m, a, c):
    """
    Exponential decay rate kappa(a) for shelf waves on background u=a.

    From Eq. (6.11) of the main paper, for c > c* = m*a^{m-1}:
        kappa = sqrt((c - m*a^{m-1}) / (m*a^{m-1}))

    Parameters
    ----------
    m : float
        Nonlinearity exponent.
    a : float
        Background level (a > 0).
    c : float
        Wave speed (must exceed c* = m*a^{m-1}).

    Returns
    -------
    kappa : float
        Exponential decay rate.
    """
    c_star = m * a ** (m - 1.0)
    if c <= c_star:
        raise ValueError(f"Speed c={c} must exceed c*={c_star}")
    return np.sqrt((c - c_star) / c_star)
