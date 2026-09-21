"""Universal formulas for K(m,m) compactons (Theorem 4.1)."""
import numpy as np
from dataclasses import dataclass

@dataclass
class CompactonProfile:
    m: float; A: float; alpha: float; ell: float; speed: float; W: float
    def evaluate(self, xi):
        U = np.zeros_like(xi, dtype=float)
        inside = np.abs(xi) <= self.W / 2
        U[inside] = self.A * np.cos(xi[inside] / self.ell) ** self.alpha
        return U

def universal_formulas(m, A=1.0):
    if m <= 1: raise ValueError(f"m must be > 1, got {m}")
    if A <= 0: raise ValueError(f"A must be > 0, got {A}")
    alpha = 2.0 / (m - 1)
    ell = 2.0 * m / (m - 1)
    speed = (m + 1) / (2 * m) * A**(m - 1)
    return CompactonProfile(m=m, A=A, alpha=alpha, ell=ell, speed=speed, W=np.pi * ell)

def support_width(m): return 2.0 * np.pi * m / (m - 1)
def speed_coefficient(m): return (m + 1) / (2.0 * m)
def vk_criterion(m, A): return (m + 1) * (m - 1) / (2 * m) * A**(m - 2)
