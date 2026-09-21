"""
Diagnostics for K(m,m) single-compacton propagation (Phase 1).

Six metrics per run:
  1. Peak amplitude drift:    |A(t_f) - A_0| / A_0
  2. Energy conservation:     |Delta I1| / I1(0)
  3. Mass conservation:       |Delta I0| / I0(0)   [expect machine precision]
  4. Noise floor L2:          ||u - U_exact||_2 outside support
  5. Shape fidelity Linf:     ||u - U_exact||_inf inside support
  6. Wake flag:               max|u| in the region behind the compacton
"""

import numpy as np
from compacton import compacton_half_support, compacton_speed, exact_solution


def conservation_laws(u, m, dx):
    """
    Compute I0 (mass) and I1 (energy) for K(m,m).

    I0 = integral u dx          (mass)
    I1 = integral |u|^{m+1}/(m+1) dx   (Hamiltonian energy proxy)

    Parameters
    ----------
    u  : ndarray  Solution field.
    m  : float    Nonlinearity exponent.
    dx : float    Grid spacing.

    Returns
    -------
    dict with 'I0' and 'I1'.
    """
    I0 = np.sum(u) * dx
    I1 = np.sum(np.abs(u)**(m + 1) / (m + 1)) * dx
    return {'I0': I0, 'I1': I1}


def compute_diagnostics(u_final, u_init, t_final, x, m, A, x0, L, dx):
    """
    Compute all six Phase 1 diagnostics.

    Parameters
    ----------
    u_final : ndarray  Numerical solution at t = t_final.
    u_init  : ndarray  Initial condition at t = 0.
    t_final : float    Final time.
    x       : ndarray  Spatial grid.
    m       : float    Nonlinearity exponent.
    A       : float    Compacton amplitude.
    x0      : float    Initial compacton center.
    L       : float    Domain length.
    dx      : float    Grid spacing.

    Returns
    -------
    dict of diagnostic values.
    """
    c = compacton_speed(m, A)
    hw = compacton_half_support(m)
    x_center = x0 + c * t_final

    # Exact solution at t_final
    u_exact = exact_solution(x, t_final, m, A=A, x0=x0, L=L)

    # --- 1. Peak amplitude drift ---
    A_final = np.max(u_final)
    amp_drift = abs(A_final - A) / A

    # --- 2 & 3. Conservation laws ---
    laws_init  = conservation_laws(u_init,  m, dx)
    laws_final = conservation_laws(u_final, m, dx)

    I0_rel = abs(laws_final['I0'] - laws_init['I0']) / (abs(laws_init['I0']) + 1e-30)
    I1_rel = abs(laws_final['I1'] - laws_init['I1']) / (abs(laws_init['I1']) + 1e-30)

    # --- 4. Noise floor L2 (outside support) ---
    # Periodic distance to compacton center
    xi = x - x_center
    xi_wrap = xi - L * np.round(xi / L)
    outside = np.abs(xi_wrap) > 2.0 * hw   # 2x half-support margin
    noise_l2 = np.sqrt(np.sum(u_final[outside]**2) * dx)

    # --- 5. Shape fidelity Linf (inside support) ---
    inside = np.abs(xi_wrap) < hw
    if np.any(inside):
        shape_linf = np.max(np.abs(u_final[inside] - u_exact[inside]))
    else:
        shape_linf = np.nan

    # --- 6. Wake flag: max|u| in the region clearly behind the compacton ---
    # "Behind" = region where the compacton has already passed
    # For rightward propagation, this is to the left of the compacton center
    # We look 2*hw to 6*hw behind the center
    wake_region = (xi_wrap < -2.0 * hw) & (xi_wrap > -6.0 * hw)
    if np.any(wake_region):
        wake_amp = np.max(np.abs(u_final[wake_region]))
    else:
        wake_amp = 0.0

    return {
        'amp_drift':   amp_drift,
        'I0_rel':      I0_rel,
        'I1_rel':      I1_rel,
        'noise_l2':    noise_l2,
        'shape_linf':  shape_linf,
        'wake_amp':    wake_amp,
        'A_final':     A_final,
        'I0_final':    laws_final['I0'],
        'I1_final':    laws_final['I1'],
        'x_center_numerical': x[np.argmax(u_final)],
        'x_center_exact':     x_center % L,
    }


def print_diagnostics(diag, m, variant, spatial):
    """Pretty-print diagnostics for a single run."""
    print(f"  K({int(m)},{int(m)}) Variant {variant} [{spatial}]:")
    print(f"    Amplitude drift : {diag['amp_drift']:.3e}")
    print(f"    Mass  |dI0|/I0  : {diag['I0_rel']:.3e}")
    print(f"    Energy|dI1|/I1  : {diag['I1_rel']:.3e}")
    print(f"    Noise floor L2  : {diag['noise_l2']:.3e}")
    print(f"    Shape fidelity  : {diag['shape_linf']:.3e}")
    print(f"    Wake amplitude  : {diag['wake_amp']:.3e}")
