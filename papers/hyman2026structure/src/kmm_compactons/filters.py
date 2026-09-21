"""
Fourier-space filter suite for K(m,n) solver.

Supports:
  - Sharp cutoff LP/HP with independent q_lp and q_hp (split-q)
  - Smooth LP filters: exponential, raised cosine, erfc
  - Spectral and FD2 dissipation operators

Convention:
    lp_mask  : applied to flux terms in RHS (cutoff at k_max / q_lp)
    hp_mask  : applied to dissipation      (active above k_max / q_hp)
    These can differ when q_lp != q_hp (split-q mode).
"""

import numpy as np
from numpy.fft import fft, ifft


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_sharp_masks(k, k_max, q_lp, q_hp=None):
    """
    Sharp-cutoff LP and HP masks with independent cutoffs.

    Parameters
    ----------
    k     : ndarray  Fourier wavenumbers.
    k_max : float    Nyquist wavenumber = pi/dx.
    q_lp  : float    LP cutoff parameter: k_c_lp = k_max / q_lp.
    q_hp  : float    HP onset parameter: k_c_hp = k_max / q_hp.
                     If None, uses q_hp = q_lp (standard: HP = 1 - LP).

    Returns
    -------
    lp_mask : ndarray  1 where |k| <= k_max/q_lp, else 0.
    hp_mask : ndarray  1 where |k| >  k_max/q_hp, else 0.
    """
    if q_hp is None:
        q_hp = q_lp
    k_c_lp = k_max / q_lp
    k_c_hp = k_max / q_hp
    lp_mask = (np.abs(k) <= k_c_lp).astype(float)
    hp_mask = (np.abs(k) >  k_c_hp).astype(float)
    return lp_mask, hp_mask


def build_smooth_lp_mask(k, k_max, q_lp, shape='exponential', p=8, q_hp=None):
    """
    Smooth LP mask with gradual roll-off, plus sharp HP mask.

    Parameters
    ----------
    k, k_max, q_lp : as in build_sharp_masks.
    shape  : 'exponential' | 'raised_cosine' | 'erfc'
    p      : order for exponential filter (default 8 → very sharp but smooth).
    q_hp   : HP onset (sharp); defaults to q_lp.

    Returns
    -------
    lp_mask : ndarray  Smooth LP weights in [0, 1].
    hp_mask : ndarray  Sharp HP mask (for dissipation).
    """
    if q_hp is None:
        q_hp = q_lp

    k_c = k_max / q_lp
    kabs = np.abs(k)

    if shape == 'exponential':
        # Vandeven-style exponential: exp(-alpha*(|k|/k_c)^{2p})
        # alpha chosen so mask = exp(-1) at |k| = k_c  →  alpha = 1
        ratio = kabs / k_c
        lp_mask = np.exp(-ratio**(2 * p))
        # Hard-zero above Nyquist boundary to prevent aliasing
        lp_mask[kabs > k_max] = 0.0

    elif shape == 'raised_cosine':
        # Smooth transition over [0.75*k_c, k_c]
        lp_mask = np.ones_like(k, dtype=float)
        k_lo = 0.75 * k_c
        trans = (kabs >= k_lo) & (kabs <= k_c)
        lp_mask[trans] = 0.5 * (1 + np.cos(np.pi * (kabs[trans] - k_lo) / (k_c - k_lo)))
        lp_mask[kabs > k_c] = 0.0

    elif shape == 'erfc':
        from scipy.special import erfc
        # Centered at k_c with width sigma = 0.1*k_c
        sigma = 0.1 * k_c
        lp_mask = 0.5 * erfc((kabs - k_c) / sigma)

    else:
        raise ValueError(f"Unknown filter shape: {shape}")

    # HP mask for dissipation remains sharp
    k_c_hp = k_max / q_hp
    hp_mask = (kabs > k_c_hp).astype(float)

    return lp_mask, hp_mask


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def apply_lp(f, lp_mask):
    """Apply LP filter (sharp or smooth) in Fourier space."""
    return np.real(ifft(lp_mask * fft(f)))


# ---------------------------------------------------------------------------
# Dissipation operators
# ---------------------------------------------------------------------------

def apply_hp_dissipation_spectral(u, k, hp_mask):
    """
    Spectral HP-filtered Laplacian: F^{-1}( -k^2 * hp_mask * F(u) ).

    Exact second derivative for all modes; strongest at high wavenumbers.
    """
    return np.real(ifft(-k**2 * hp_mask * fft(u)))


def apply_hp_dissipation_fd2(u, dx, hp_mask, k):
    """
    FD2 second derivative with HP filtering in Fourier space.

    Computes the 3-point FD2 stencil (u_{j+1} - 2u_j + u_{j-1})/dx^2,
    then applies the HP mask in Fourier space.

    The FD2 modified wavenumber (2-2cos(k*dx))/dx^2 tapers to zero at
    Nyquist, providing softer high-frequency dissipation than spectral -k^2.
    This may better preserve structure near the compacton support boundary.
    """
    # FD2 second derivative
    u_xx_fd2 = (np.roll(u, -1) - 2.0*u + np.roll(u, 1)) / dx**2
    # HP filter in Fourier space
    return np.real(ifft(hp_mask * fft(u_xx_fd2)))


# ---------------------------------------------------------------------------
# Smooth HP ramp mask
# ---------------------------------------------------------------------------

def build_smooth_hp_ramp(k, k_max, q_hp):
    """
    Smooth linear ramp HP mask: 0 at k_max/q_hp, 1 at k_max.

    r(k) = (|k| - k_c) / (k_max - k_c)  for k_c <= |k| <= k_max
           0                              for |k| < k_c
           1                              for |k| > k_max (safety)

    Combined with the -k^2 or k^4 operator, this gives a smooth onset
    of dissipation rather than an abrupt step, avoiding Gibbs ringing
    from the dissipation operator itself.

    Parameters
    ----------
    k     : ndarray  Fourier wavenumbers.
    k_max : float    Nyquist wavenumber.
    q_hp  : float    Ramp onset: k_c = k_max / q_hp.

    Returns
    -------
    ramp : ndarray  Smooth HP ramp weights in [0, 1].
    """
    k_c  = k_max / q_hp
    kabs = np.abs(k)
    ramp = np.zeros_like(k, dtype=float)
    in_ramp = (kabs >= k_c) & (kabs <= k_max)
    ramp[in_ramp] = (kabs[in_ramp] - k_c) / (k_max - k_c)
    ramp[kabs > k_max] = 1.0
    return ramp


# ---------------------------------------------------------------------------
# Biharmonic dissipation  (scaling: dx^3 to keep delta dimensionless)
# ---------------------------------------------------------------------------

def apply_biharmonic_dissipation(u, k, hp_mask, dx, ramp=None):
    """
    Spectral biharmonic HP dissipation.

    Returns F^{-1}( k^4 * W(k) * F(u) )
    scaled by dx^3 (not dx) so that delta remains dimensionless
    with the same O(1) range as the Laplacian dissipation.

    At Nyquist: k^4 * dx^3 = (pi/dx)^4 * dx^3 = pi^4 / dx
    matches the Laplacian scaling: k^2 * dx = (pi/dx)^2 * dx = pi^2 / dx
    up to the factor pi^2 ≈ 9.87 (accounted for by using smaller delta).

    Parameters
    ----------
    u      : ndarray  Solution field.
    k      : ndarray  Fourier wavenumbers.
    hp_mask: ndarray  Sharp or smooth HP mask (step or ramp).
    dx     : float    Grid spacing (for dx^3 scaling factor).
    ramp   : ndarray or None
        If provided, replaces hp_mask with the smooth ramp weights.
        Pass result of build_smooth_hp_ramp() here.

    Returns
    -------
    result : ndarray  Biharmonic dissipation field (before delta scaling).
    """
    mask = ramp if ramp is not None else hp_mask
    # Negative sign: -(-Δ)^2 = -k^4 is dissipative (removes energy)
    return np.real(ifft(-k**4 * mask * fft(u)))


def apply_laplacian_dissipation(u, k, hp_mask, ramp=None):
    """
    Spectral Laplacian HP dissipation (renamed for clarity alongside biharmonic).

    Returns F^{-1}( -k^2 * W(k) * F(u) )
    scaled by dx externally (unchanged from original apply_hp_dissipation_spectral).

    Parameters
    ----------
    ramp : ndarray or None  If provided, uses smooth ramp instead of sharp step.
    """
    mask = ramp if ramp is not None else hp_mask
    return np.real(ifft(-k**2 * mask * fft(u)))
