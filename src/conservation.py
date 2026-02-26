"""
Conservation laws for the K(m,m) equation.

The K(m,m) equation possesses exactly five known independent conservation
laws (Vodová 2013, Proposition 4.3 of the paper):

  I_0 = int u dx                       (mass)
  I_1 = int u^{m+1} dx                 (energy, also Hamiltonian)
  I_2 = int u sin(x) dx                (trigonometric)
  I_3 = int u cos(x) dx                (trigonometric)
  I_4 = int u (D_xx + 1)^{-1} u dx     (nonlocal momentum)

The first four are local; I_4 is nonlocal and involves the pseudoinverse
of the Helmholtz operator restricted to mean-zero functions.

Reference: Section 4.2 and Theorem 4.6 of the Lax Centennial paper.
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq


def mass(u, dx):
    """I_0 = integral of u dx."""
    return np.sum(u) * dx


def energy(u, m, dx):
    """I_1 = integral of u^{m+1} dx."""
    return np.sum(np.sign(u) * np.abs(u) ** (m + 1)) * dx


def trig_sin(u, x, dx):
    """I_2 = integral of u sin(x) dx."""
    return np.sum(u * np.sin(x)) * dx


def trig_cos(u, x, dx):
    """I_3 = integral of u cos(x) dx."""
    return np.sum(u * np.cos(x)) * dx


def nonlocal_momentum(u, x, dx):
    """
    I_4 = integral of u * (D_xx + 1)^{-1} u dx.

    Computed via Fourier: (D_xx + 1)^{-1} has symbol 1/(1 - k^2)
    for k != ±1, and the pseudoinverse sets the k=±1 modes to zero.

    Note: This is the pseudoinverse formulation (Section 4.2.3),
    not the exponential-kernel form. The L^2 duality pairing
    makes this well-defined for compactons since they have
    I_2 = I_3 = 0 (Proposition 4.8).
    """
    N = len(u)
    L = N * dx
    k = fftfreq(N, dx) * 2 * np.pi
    u_hat = fft(u)

    # Pseudoinverse: set k=±1 modes to zero
    symbol = np.zeros(N, dtype=complex)
    for i in range(N):
        denom = 1.0 - k[i] ** 2
        if abs(denom) > 1e-10:
            symbol[i] = 1.0 / denom
        # else: pseudoinverse sets to 0

    v = np.real(ifft(symbol * u_hat))
    return np.sum(u * v) * dx


def all_conservation_laws(u, x, m):
    """
    Evaluate all five conservation laws.

    Parameters
    ----------
    u : ndarray
        Solution field.
    x : ndarray
        Spatial grid.
    m : float
        Nonlinearity exponent.

    Returns
    -------
    dict with keys I0, I1, I2, I3, I4.
    """
    dx = x[1] - x[0]
    return {
        'I0': mass(u, dx),
        'I1': energy(u, m, dx),
        'I2': trig_sin(u, x, dx),
        'I3': trig_cos(u, x, dx),
        'I4': nonlocal_momentum(u, x, dx),
    }


def conservation_drift(laws_initial, laws_final):
    """Compute relative drift in each conservation law."""
    drift = {}
    for key in laws_initial:
        val0 = laws_initial[key]
        val1 = laws_final[key]
        if abs(val0) > 1e-30:
            drift[key] = abs(val1 - val0) / abs(val0)
        else:
            drift[key] = abs(val1 - val0)
    return drift
