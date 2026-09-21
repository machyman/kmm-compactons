"""Shelf-wave analysis (Section 7: critical speed, Theorem 7.1; compacton singular limit, Theorem 7.6)."""
import numpy as np
from scipy.integrate import solve_ivp

def critical_speed(m, a): return m * a**(m - 1)
def decay_rate(m, a, c):
    c_star = critical_speed(m, a)
    if c <= c_star: raise ValueError(f"c={c} must exceed c*={c_star}")
    return np.sqrt(c / c_star - 1)

def compacton_limit_k22(c, N=1001):
    xi = np.linspace(-2*np.pi, 2*np.pi, N)
    return xi, (4*c/3) * np.cos(xi/4)**2
