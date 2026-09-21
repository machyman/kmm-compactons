"""
Spatial derivative operators for K(m,n) solver.

Two families:
  FD4  — 4th-order central finite differences (periodic BCs via np.roll)
  Spectral — exact for resolved modes via Fourier multiplication

Both return the derivative on the same spatial grid as input.

Stencil coefficients:
  D1 (4th-order, 5-point):   [-1, 8, 0, -8, 1] / (12*dx)
  D3 (4th-order, 7-point):   [1, -8, 13, 0, -13, 8, -1] / (8*dx^3)
"""

import numpy as np
from numpy.fft import fft, ifft


# ---------------------------------------------------------------------------
# FD4 stencils
# ---------------------------------------------------------------------------

def fd4_d1(f, dx):
    """
    4th-order central first derivative (5-point stencil), periodic BCs.

    (f_x)_j = (-f_{j+2} + 8*f_{j+1} - 8*f_{j-1} + f_{j-2}) / (12*dx)
    """
    return (
        -np.roll(f, -2)
        + 8.0 * np.roll(f, -1)
        - 8.0 * np.roll(f, 1)
        + np.roll(f, 2)
    ) / (12.0 * dx)


def fd4_d3(f, dx):
    """
    4th-order central third derivative (7-point stencil), periodic BCs.

    (f_xxx)_j = (-f_{j+3} + 8*f_{j+2} - 13*f_{j+1}
                 + 13*f_{j-1} - 8*f_{j-2} + f_{j-3}) / (8*dx^3)
    """
    return (
        -np.roll(f, -3)
        + 8.0 * np.roll(f, -2)
        - 13.0 * np.roll(f, -1)
        + 13.0 * np.roll(f, 1)
        - 8.0 * np.roll(f, 2)
        + np.roll(f, 3)
    ) / (8.0 * dx**3)


# ---------------------------------------------------------------------------
# Spectral derivative operators
# ---------------------------------------------------------------------------

def spec_d1(f, k):
    """
    Spectral first derivative: F^{-1}( i*k * F(f) ).

    Exact for all resolved Fourier modes.
    """
    return np.real(ifft(1j * k * fft(f)))


def spec_d3(f, k):
    """
    Spectral third derivative: F^{-1}( (i*k)^3 * F(f) ) = F^{-1}( -i*k^3 * F(f) ).

    Exact for all resolved Fourier modes.
    """
    return np.real(ifft((1j * k)**3 * fft(f)))


# ---------------------------------------------------------------------------
# Sign-preserving nonlinear power  u^m  (avoids complex values for non-integer m)
# ---------------------------------------------------------------------------

def nonlinear_power(u, m):
    """
    Compute sign-preserving u^m: sign(u) * |u|^m.

    Required for non-integer m and for negative u values.
    For the K(m,m) compacton with u >= 0 initial data, u remains >= 0,
    but this is included for robustness.
    """
    return np.sign(u) * np.abs(u)**m
