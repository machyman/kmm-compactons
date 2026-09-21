"""Conservation law evaluation for K(m,m) compactons (Theorem 4.6; trigonometric orthogonality, Proposition 5.3)."""
import numpy as np
from scipy.special import beta

def evaluate_conservation_laws(u, x, m):
    dx = x[1] - x[0]
    return {
        'I0': np.trapezoid(u, dx=dx),
        'I1': np.trapezoid(u**(m + 1) / (m + 1), dx=dx),
        'I2': np.trapezoid(u * np.sin(x), dx=dx),
        'I3': np.trapezoid(u * np.cos(x), dx=dx),
    }

def verify_trig_orthogonality(m, A=1.0, N=4096):
    from .compacton import universal_formulas
    p = universal_formulas(m, A)
    x = np.linspace(-3 * p.W, 3 * p.W, N)
    u = p.evaluate(x); dx = x[1] - x[0]
    return {
        'I2': np.trapezoid(u * np.sin(x), dx=dx),
        'I3': np.trapezoid(u * np.cos(x), dx=dx),
        'ell_minus_alpha': p.ell - p.alpha,
    }
