"""Shared utilities."""
import numpy as np
def l2_norm(u, dx): return np.sqrt(np.trapezoid(u**2, dx=dx))
def relative_error(computed, reference):
    return abs(computed - reference) / abs(reference) if abs(reference) > 1e-30 else abs(computed)
