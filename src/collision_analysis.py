"""
Post-collision analysis for compacton and soliton collisions.

Provides:
  - Amplitude extraction via Nelder-Mead fitting
  - Debris field isolation
  - Spectral analysis (FFT-based)
  - Spectral notch detection at k=1
  - SNR estimation against single-propagation noise floor

Reference: Table 4 and Section 5.1 of the Lax Centennial paper.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.fft import fft, fftfreq

try:
    from .compacton_physics import compacton_profile
except ImportError:
    from compacton_physics import compacton_profile


def extract_debris(u_post, m, x):
    """
    Extract debris field by fitting and subtracting two compactons.

    Parameters
    ----------
    u_post : ndarray
        Post-collision solution.
    m : float
        Nonlinearity exponent.
    x : ndarray
        Spatial grid.

    Returns
    -------
    debris : ndarray
        Residual debris field.
    A1_fit, x1_fit, A2_fit, x2_fit : float
        Fitted compacton amplitudes and positions.
    fit_residual : float
        L2 fitting residual.
    """
    dx = x[1] - x[0]
    peaks, _ = find_peaks(u_post, height=0.02, distance=10)
    if len(peaks) < 2:
        return u_post, 0, 0, 0, 0, np.inf
    heights = u_post[peaks]
    idx_sort = np.argsort(heights)[::-1]
    p1, p2 = peaks[idx_sort[0]], peaks[idx_sort[1]]

    def residual(params):
        a1, x01, a2, x02 = params
        if a1 < 0.01 or a2 < 0.01:
            return 1e10
        fit = compacton_profile(m, a1, x, x01) + compacton_profile(m, a2, x, x02)
        return np.sum((u_post - fit) ** 2) * dx

    res = minimize(residual, [u_post[p1], x[p1], u_post[p2], x[p2]],
                   method='Nelder-Mead',
                   options={'xatol': 1e-10, 'fatol': 1e-16, 'maxiter': 10000})
    A1f, x1f, A2f, x2f = res.x
    fitted = compacton_profile(m, A1f, x, x1f) + compacton_profile(m, A2f, x, x2f)
    return u_post - fitted, A1f, x1f, A2f, x2f, res.fun


def extract_single_noise(u_post, m, x):
    """Extract noise from single-compacton propagation control."""
    dx = x[1] - x[0]
    pk = np.argmax(u_post)
    res = minimize(
        lambda p: np.sum((u_post - compacton_profile(m, p[0], x, p[1])) ** 2) * dx,
        [u_post[pk], x[pk]], method='Nelder-Mead',
        options={'xatol': 1e-10, 'fatol': 1e-16, 'maxiter': 10000})
    fitted = compacton_profile(m, res.x[0], x, res.x[1])
    return u_post - fitted, res.x[0], res.x[1]


def get_spectrum(signal, x):
    """Compute one-sided Fourier amplitude spectrum."""
    N = len(x)
    dx = x[1] - x[0]
    s_hat = fft(signal)
    kwave = fftfreq(N, dx) * 2 * np.pi
    pos = kwave >= 0
    kp, sp = kwave[pos], np.abs(s_hat[pos])
    si = np.argsort(kp)
    return kp[si], sp[si]


def analyze_below_cutoff(k, spec, k_cut_phys, k_target=1.0):
    """
    Analyze spectrum below the filter cutoff.

    Returns
    -------
    dict with keys:
        w_k1 : float — spectral amplitude at k=1
        k_peak, s_peak : float — peak wavenumber and amplitude
        notch_ratio : float — |w_hat(1)| / max|w_hat|
        spec_L2 : float — L2 norm of below-cutoff spectrum
    """
    below = k < k_cut_phys
    k_b, s_b = k[below], spec[below]
    if len(k_b) == 0:
        return {'w_k1': 0, 'k_peak': 0, 's_peak': 0, 'notch_ratio': 999,
                'spec_L2': 0, 'k_below': k_b, 'spec_below': s_b}

    k1_idx = np.argmin(np.abs(k_b - k_target))
    w_k1 = s_b[k1_idx]
    valid = k_b > 0.1
    if np.any(valid):
        pi = np.argmax(s_b[valid])
        k_peak, s_peak = k_b[valid][pi], s_b[valid][pi]
    else:
        k_peak, s_peak = 0.0, 0.0
    notch = w_k1 / s_peak if s_peak > 0 else 999.0
    dk = k_b[1] - k_b[0] if len(k_b) > 1 else 1.0
    spec_L2 = np.sqrt(np.sum(s_b ** 2) * dk)
    return {'w_k1': w_k1, 'k_peak': k_peak, 's_peak': s_peak,
            'notch_ratio': notch, 'spec_L2': spec_L2,
            'k_below': k_b, 'spec_below': s_b}


def elasticity_coefficient(A1_pre, A2_pre, A1_post, A2_post):
    """Amplitude elasticity: Eq. (31) of the paper."""
    return (A1_post + A2_post) / (A1_pre + A2_pre)
