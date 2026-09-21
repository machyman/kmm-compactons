"""
KMNSolver: spectrally-filtered solver for K(m,m) compacton equations.

Conservative form:
    u_t + (u^m)_x + (u^m)_{xxx} = delta * c_max * dx^p * DISS[u]

where c_max = (m+1)/(2m)*max|u|^{m-1} is the max wave speed (v2, 2026-03-04).
[v1 used delta*max|u|^{m-1} without the (m+1)/(2m) factor.]

where p=1 for Laplacian (diss_power=2) and p=3 for biharmonic (diss_power=4).

Filter variants:
    A: F_LP[(u^m)_x]  + (u^m)_{xxx}
    C: F_LP[(u^m)_x   + (u^m)_{xxx}]

Dissipation parameters:
    diss_power  : 2 (Laplacian, -k^2, dx scaling) or
                  4 (biharmonic, k^4, dx^3 scaling)
    hp_ramp     : False (sharp step HP mask) or
                  True  (smooth linear ramp from 0 at k_c to 1 at k_max)
"""

import numpy as np
from numpy.fft import fftfreq
from scipy.integrate import solve_ivp

from filters import (build_sharp_masks, build_smooth_lp_mask,
                     build_smooth_hp_ramp,
                     apply_lp,
                     apply_hp_dissipation_spectral,
                     apply_hp_dissipation_fd2,
                     apply_laplacian_dissipation,
                     apply_biharmonic_dissipation)
from stencils import fd4_d1, fd4_d3, spec_d1, spec_d3, nonlinear_power
from compacton_profiles import compacton_speed


class KMNSolver:
    """
    Spectrally-filtered K(m,m) solver.

    Parameters
    ----------
    m            : float  Nonlinearity exponent (m > 1).
    Nx           : int    Grid points (default 512).
    L            : float  Domain length (default 150).
    delta        : float  Dissipation coefficient (default 0.5).
    q_lp         : float  LP filter cutoff k_max/q_lp (default 2).
    q_hp         : float  HP dissipation onset k_max/q_hp (default=q_lp).
    variant      : str    'A' or 'C'.
    spatial      : str    'fd4' or 'spectral'.
    filter_shape : str    'sharp' | 'exponential' | 'raised_cosine' | 'erfc'.
    filter_p     : int    Exponential filter order (default 8).
    diss_method  : str    'spectral' | 'fd2'  (only for Laplacian power).
    diss_power   : int    2 (Laplacian, dx scaling) or 4 (biharmonic, dx^3 scaling).
    hp_ramp      : bool   True = smooth linear ramp HP mask; False = sharp step.
    """

    def __init__(self, m, Nx=512, L=150.0, delta=0.5,
                 q_lp=2.0, q_hp=None,
                 variant='C', spatial='fd4',
                 filter_shape='sharp', filter_p=8,
                 diss_method='spectral',
                 diss_power=2,
                 hp_ramp=False):

        self.m            = float(m)
        self.Nx           = Nx
        self.L            = L
        self.delta        = delta
        self.q_lp         = float(q_lp)
        self.q_hp         = float(q_hp) if q_hp is not None else float(q_lp)
        self.variant      = variant.upper()
        self.spatial      = spatial.lower()
        self.filter_shape = filter_shape
        self.filter_p     = filter_p
        self.diss_method  = diss_method.lower()
        self.diss_power   = diss_power
        self.hp_ramp      = hp_ramp

        if self.variant not in ('A', 'C'):
            raise ValueError(f"variant must be 'A' or 'C', got '{variant}'")
        if self.spatial not in ('fd4', 'spectral'):
            raise ValueError(f"spatial must be 'fd4' or 'spectral'")
        if self.diss_method not in ('spectral', 'fd2'):
            raise ValueError(f"diss_method must be 'spectral' or 'fd2'")
        if self.diss_power not in (2, 4):
            raise ValueError(f"diss_power must be 2 or 4")

        # Grid
        self.dx = L / Nx
        self.x  = np.linspace(0.0, L - self.dx, Nx)

        # Fourier wavenumbers
        self.k     = 2.0 * np.pi * fftfreq(Nx, d=self.dx)
        self.k_max = np.pi / self.dx

        # Build filter masks
        self._build_masks()

        # CFL timestep (dispersive: dt ~ C * dx^3 / m)
        self._cfl_safety = 0.8 * 2.83 / 4.0
        self.dt_cfl = self._cfl_safety * self.dx**3 / self.m

    def _build_masks(self):
        if self.filter_shape == 'sharp':
            self.lp_mask, self.hp_mask = build_sharp_masks(
                self.k, self.k_max, self.q_lp, self.q_hp)
        else:
            self.lp_mask, self.hp_mask = build_smooth_lp_mask(
                self.k, self.k_max, self.q_lp,
                shape=self.filter_shape, p=self.filter_p,
                q_hp=self.q_hp)

        # Build smooth ramp if requested
        if self.hp_ramp:
            self.hp_ramp_weights = build_smooth_hp_ramp(
                self.k, self.k_max, self.q_hp)
        else:
            self.hp_ramp_weights = None

    # ------------------------------------------------------------------
    # Derivative dispatch
    # ------------------------------------------------------------------

    def _d1(self, f):
        return fd4_d1(f, self.dx) if self.spatial == 'fd4' else spec_d1(f, self.k)

    def _d3(self, f):
        return fd4_d3(f, self.dx) if self.spatial == 'fd4' else spec_d3(f, self.k)

    # ------------------------------------------------------------------
    # Dissipation dispatch
    # ------------------------------------------------------------------

    def _dissipation(self, u):
        max_u = np.max(np.abs(u)) + 1e-30
        ramp  = self.hp_ramp_weights  # None if sharp step

        # c_max scaling: delta is now nondimensionalised by the true maximum
        # wave speed c_max = (m+1)/(2m) * max|u|^{m-1}, making the optimal
        # delta independent of m (boundary-layer argument, Hyman 2026).
        c_max_factor = (self.m + 1) / (2.0 * self.m)

        if self.diss_power == 4:
            # Biharmonic: scale by dx^3 to keep delta dimensionless
            scale = self.delta * c_max_factor * max_u**(self.m - 1) * self.dx**3
            return scale * apply_biharmonic_dissipation(
                u, self.k, self.hp_mask, self.dx, ramp=ramp)
        else:
            # Laplacian: scale by dx
            scale = self.delta * c_max_factor * max_u**(self.m - 1) * self.dx
            if self.diss_method == 'fd2':
                return scale * apply_hp_dissipation_fd2(
                    u, self.dx, self.hp_mask, self.k)
            else:
                return scale * apply_laplacian_dissipation(
                    u, self.k, self.hp_mask, ramp=ramp)

    # ------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------

    def rhs(self, u):
        flux     = nonlinear_power(u, self.m)
        flux_x   = self._d1(flux)
        flux_xxx = self._d3(flux)

        if self.variant == 'A':
            advection  = apply_lp(flux_x, self.lp_mask)
            dispersion = flux_xxx
            return -advection - dispersion + self._dissipation(u)
        else:  # 'C'
            combined = apply_lp(flux_x + flux_xxx, self.lp_mask)
            return -combined + self._dissipation(u)

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def estimate_dt(self, u0, t_probe=5.0, rtol=1e-8, atol=1e-10):
        """
        Run RKF45 adaptive integration for t_probe to find a reliable fixed dt.

        Extracts internal adaptive step sizes from the dense_output object.
        Returns safety * median_adaptive_step, capped at dt_cfl.
        """
        safety = 0.8

        sol = solve_ivp(
            lambda t, y: self.rhs(y),
            [0, t_probe], u0,
            method='RK45',
            t_eval=np.linspace(0, t_probe, 2000),
            rtol=rtol, atol=atol,
            dense_output=True
        )

        if not sol.success:
            dt = self.dt_cfl * 0.5
            return dt, {'dt_cfl': self.dt_cfl, 'dt_recommended': dt,
                        'probe_success': False, 'n_steps': None,
                        'dt_median': None, 'dt_min': None}

        try:
            ts = np.array(sol.sol.ts)
            internal_steps = np.diff(ts)
            internal_steps = internal_steps[internal_steps > 0]
            dt_median = float(np.median(internal_steps))
            dt_min    = float(np.min(internal_steps))
            n_steps   = len(internal_steps)
            dt_recommended = max(safety * dt_median, self.dt_cfl / 16.0)
            dt_recommended = min(dt_recommended, self.dt_cfl)
        except (AttributeError, TypeError):
            dt_recommended = self.dt_cfl * 0.9
            dt_median = dt_recommended
            dt_min    = dt_recommended
            n_steps   = None

        info = {
            'dt_cfl':         self.dt_cfl,
            'dt_recommended': dt_recommended,
            'dt_median':      dt_median,
            'dt_min':         dt_min,
            'probe_success':  True,
            'n_steps':        n_steps,
        }
        return dt_recommended, info

    def step_rk4(self, u, dt):
        k1 = self.rhs(u)
        k2 = self.rhs(u + 0.5*dt*k1)
        k3 = self.rhs(u + 0.5*dt*k2)
        k4 = self.rhs(u + dt*k3)
        return u + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

    def solve(self, u0, t_final, dt, save_interval=10.0, verbose=False):
        """Fixed-step RK4 production run."""
        n_steps = int(np.ceil(t_final / dt))
        dt_adj  = t_final / n_steps

        u = u0.copy()
        t = 0.0
        t_history = [0.0]
        u_history = [u.copy()]
        next_save = save_interval

        for _ in range(n_steps):
            u = self.step_rk4(u, dt_adj)
            t += dt_adj
            if t >= next_save - 0.5*dt_adj:
                t_history.append(float(t))
                u_history.append(u.copy())
                next_save += save_interval
                if verbose:
                    print(f"  t={t:.1f}")

        if abs(t_history[-1] - t_final) > 0.5*dt_adj:
            t_history.append(t_final)
            u_history.append(u.copy())

        return t_history, u_history
