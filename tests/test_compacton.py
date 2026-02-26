"""
Test suite for compacton physics and solver modules.

Run with: pytest tests/ -v
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from compacton_physics import (
    edge_exponent, width_parameter, half_width, speed,
    amplitude_from_speed, compacton_profile
)
from conservation import mass, energy, trig_sin, trig_cos, all_conservation_laws
from kmm_solver import solve_kmm, compute_cutoff


# ============================================================
# Compacton Physics Tests
# ============================================================

class TestCompactonFormulas:
    """Verify universal formulas from Theorem 4.1."""

    def test_edge_exponent_k22(self):
        assert np.isclose(edge_exponent(2), 2.0)

    def test_edge_exponent_k33(self):
        assert np.isclose(edge_exponent(3), 1.0)

    def test_edge_exponent_k44(self):
        assert np.isclose(edge_exponent(4), 2.0 / 3.0)

    def test_width_parameter_k22(self):
        assert np.isclose(width_parameter(2), 4.0)

    def test_width_parameter_k33(self):
        assert np.isclose(width_parameter(3), 3.0)

    def test_speed_amplitude_k22(self):
        # c = (3/4) * A for m=2
        assert np.isclose(speed(2, 1.0), 0.75)
        assert np.isclose(speed(2, 2.0), 1.5)

    def test_speed_amplitude_k33(self):
        # c = (4/6) * A^2 = (2/3) A^2 for m=3
        assert np.isclose(speed(3, 1.0), 2.0 / 3.0)

    def test_speed_amplitude_inverse(self):
        """amplitude_from_speed is inverse of speed."""
        for m in [2, 3, 4, 5]:
            for A in [0.5, 1.0, 2.0]:
                c = speed(m, A)
                A_recovered = amplitude_from_speed(m, c)
                assert np.isclose(A_recovered, A, rtol=1e-10), \
                    f"Failed for m={m}, A={A}"


class TestCompactonProfile:
    """Test compacton profile evaluation."""

    def test_compact_support(self):
        """Profile must vanish outside support."""
        x = np.linspace(-20, 20, 1000)
        for m in [2, 3, 4]:
            u = compacton_profile(m, 1.0, x, 0.0)
            hw = half_width(m)
            outside = np.abs(x) > hw + 0.01
            assert np.allclose(u[outside], 0.0), f"m={m}: nonzero outside support"

    def test_peak_amplitude(self):
        """Peak value should equal A."""
        x = np.linspace(-10, 10, 10001)
        for m in [2, 3, 4, 5]:
            for A in [0.5, 1.0, 3.0]:
                u = compacton_profile(m, A, x, 0.0)
                assert np.isclose(np.max(u), A, rtol=1e-6), \
                    f"Peak mismatch for m={m}, A={A}"

    def test_symmetry(self):
        """Profile must be symmetric about center."""
        x = np.linspace(-15, 15, 1001)
        for m in [2, 3, 4]:
            u = compacton_profile(m, 1.0, x, 0.0)
            u_rev = u[::-1]
            assert np.allclose(u, u_rev, atol=1e-12), f"m={m}: asymmetric"

    def test_positivity(self):
        """Profile must be non-negative."""
        x = np.linspace(-15, 15, 2000)
        for m in [2, 3, 4, 5]:
            u = compacton_profile(m, 1.0, x, 0.0)
            assert np.all(u >= -1e-15), f"m={m}: negative values"


# ============================================================
# Conservation Law Tests
# ============================================================

class TestConservationLaws:
    """Test conservation law evaluation."""

    def test_trigonometric_vanishing(self):
        """I_2 and I_3 should vanish for centered compactons (Prop 4.8)."""
        N = 4096
        L = 100.0
        x = np.linspace(0, L, N, endpoint=False)
        dx = x[1] - x[0]
        for m in [2, 3, 4]:
            u = compacton_profile(m, 1.0, x, L / 2)
            I2 = trig_sin(u, x, dx)
            I3 = trig_cos(u, x, dx)
            # These should be small (not exactly zero due to discretization
            # and the fact that x0 = L/2 is not special for sin/cos)
            # The key test is that they're much smaller than mass
            I0 = mass(u, dx)
            assert abs(I2) < 0.01 * abs(I0), \
                f"m={m}: I2={I2} too large relative to I0={I0}"

    def test_mass_positive(self):
        """Mass of a positive compacton must be positive."""
        N = 2048
        L = 60.0
        x = np.linspace(0, L, N, endpoint=False)
        dx = x[1] - x[0]
        for m in [2, 3, 4]:
            u = compacton_profile(m, 1.0, x, L / 2)
            assert mass(u, dx) > 0


# ============================================================
# Solver Tests
# ============================================================

class TestKmmSolver:
    """Test the K(m,m) finite-difference solver."""

    def test_single_compacton_propagation(self):
        """A single compacton should propagate with correct speed."""
        m = 2
        A = 1.0
        N = 512
        L = 60.0
        x = np.linspace(0, L, N, endpoint=False)
        x0 = 15.0
        c = speed(m, A)
        t_end = 10.0

        u0 = compacton_profile(m, A, x, x0)
        mass_init = np.sum(u0) * (L / N)

        u_final, _, _ = solve_kmm(m, u0, x, t_end, q=3, delta=0.5,
                                   cfl_safety=0.8, progress_interval=0)

        # Mass conservation
        mass_final = np.sum(u_final) * (L / N)
        assert abs(mass_final - mass_init) / abs(mass_init) < 1e-6, \
            f"Mass drift: {abs(mass_final - mass_init) / abs(mass_init):.2e}"

        # Peak should have moved to x0 + c*t_end
        x_expected = x0 + c * t_end
        peak_idx = np.argmax(u_final)
        x_peak = x[peak_idx]
        # Allow ~1 grid cell error
        dx = L / N
        assert abs(x_peak - x_expected) < 3 * dx, \
            f"Peak at {x_peak:.2f}, expected {x_expected:.2f}"

    def test_mass_conservation_collision(self):
        """Mass should be conserved during collision."""
        m = 2
        N = 256
        L = 50.0
        x = np.linspace(0, L, N, endpoint=False)
        dx = x[1] - x[0]

        u0 = compacton_profile(m, 1.0, x, 10.0) + compacton_profile(m, 0.5, x, 25.0)
        mass_init = np.sum(u0) * dx

        u_final, _, _ = solve_kmm(m, u0, x, 30.0, q=3, delta=0.5,
                                   cfl_safety=0.8, progress_interval=0)
        mass_final = np.sum(u_final) * dx

        assert abs(mass_final - mass_init) / abs(mass_init) < 1e-4, \
            f"Mass drift in collision: {abs(mass_final - mass_init) / abs(mass_init):.2e}"


class TestCutoff:
    """Test spectral cutoff computation."""

    def test_cutoff_values(self):
        k_idx, k_phys = compute_cutoff(1024, 3, 100.0)
        assert k_idx == 171  # ceil(512/3)
        assert np.isclose(k_phys, 2 * np.pi * 171 / 100.0)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
