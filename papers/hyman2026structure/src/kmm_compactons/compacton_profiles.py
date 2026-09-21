"""
Exact K(m,m) compacton traveling wave solutions.

The K(m,m) equation is:
    u_t + (u^m)_x + (u^m)_{xxx} = 0

Exact compacton solution (Rosenau & Hyman 1993):

    U(x, t) = A * cos^alpha( (x - c*t) / ell )  for |x - c*t| < pi*ell/2
             = 0                                  otherwise

where:
    alpha = 2 / (m - 1)
    ell   = 2*m / (m - 1)
    c     = (m+1)/(2m) * A^{m-1}

Phase 1: single-compacton propagation only.
"""

import numpy as np


def edge_exponent(m):
    """Profile exponent alpha = 2/(m-1)."""
    return 2.0 / (m - 1)


def compacton_width(m):
    """Width parameter ell = 2m/(m-1)."""
    return 2.0 * m / (m - 1)


def compacton_half_support(m):
    """Half-support radius = pi * ell / 2."""
    return np.pi * compacton_width(m) / 2.0


def compacton_speed(m, A=1.0):
    """
    Speed-amplitude relation: c = (m+1)/(2m) * A^{m-1}.

    At A=1:
        m=2: c = 3/4
        m=3: c = 2/3
        m=4: c = 5/8
        m=5: c = 3/5
    """
    return (m + 1) / (2.0 * m) * A ** (m - 1)


def compacton_profile(x, m, A=1.0, x0=0.0, L=None):
    """
    Exact K(m,m) compacton profile centered at x0.

    Parameters
    ----------
    x  : ndarray  Spatial grid.
    m  : float    Nonlinearity exponent (m > 1).
    A  : float    Amplitude.
    x0 : float    Center position.
    L  : float or None
        Domain length for periodic wrap.

    Returns
    -------
    u : ndarray
    """
    alpha = edge_exponent(m)
    ell = compacton_width(m)
    half_w = compacton_half_support(m)

    xi = x - x0
    if L is not None:
        xi = xi - L * np.round(xi / L)

    u = np.zeros_like(x, dtype=float)
    mask = np.abs(xi) < half_w
    u[mask] = A * np.cos(xi[mask] / ell) ** alpha
    return u


def exact_solution(x, t, m, A=1.0, x0=0.0, L=None):
    """Exact solution at time t: compacton centered at x0 + c*t."""
    c = compacton_speed(m, A)
    return compacton_profile(x, m, A=A, x0=x0 + c * t, L=L)


def compacton_summary(m, A=1.0):
    """Print compacton parameters."""
    alpha = edge_exponent(m)
    ell = compacton_width(m)
    hw = compacton_half_support(m)
    c = compacton_speed(m, A)
    smoothness = {2: "C^1", 3: "C^0 (corner)", 4: "Holder C^{0,2/3}", 5: "Holder C^{0,1/2}"}
    print(f"K({m},{m}) A={A}: alpha={alpha:.4f}, ell={ell:.4f}, half-support={hw:.4f}, c={c:.6f}, x(t=100)={c*100:.2f}")
    print(f"  Smoothness: {smoothness.get(int(m), 'see alpha')}")
