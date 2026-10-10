# kmm-compactons

**Computational tools for the K(m,m) compacton equations**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/machyman/kmm-compactons/blob/main/notebooks/production_multi_m_D3v4.ipynb)

Companion code for:

> **What Remains When Integrability Fails? Mathematical Structure in the K(m,m) Compacton Equations**
> James M. Hyman
> *Discrete and Continuous Dynamical Systems — Series A*, Peter D. Lax Centennial Memorial Volume (2026)

**Reading the paper?** Every file the paper and its Supplement name, and the scripts that reproduce §5.1, §6 and Appendix C, are indexed in [papers/hyman2026structure/](papers/hyman2026structure/README.md).

---

## Overview

The K(m,m) equation

$$u_t + (u^m)_x + (u^m)_{xxx} = 0, \quad m > 1$$

is a nonlinearly dispersive PDE admitting *compactons* — solitary waves with compact support.  Despite failing every standard integrability test, K(m,m) exhibits strikingly near-elastic collisions.  This repository provides the computational tools used to investigate the mathematical structure that persists in the absence of integrability.

### What's in this repository

| Component | Description |
|-----------|-------------|
| **K(m,m) solvers** | Pseudospectral (PSM) solver, the method of every K(m,m) collision run in the paper (Appendix D): package `papers/hyman2026structure/src/kmm_compactons/`, behind §5.1 and Figure 3 ([index](papers/hyman2026structure/README.md#the-pseudospectral-package)). Regularized finite-difference solver with dual spectral filtering and RK4 (`src/kmm_solver.py`), the scheme of the cross-check notebook |
| **BBM solver** | Pseudospectral solver for comparison experiments |
| **Collision analysis** | Post-collision amplitude extraction, debris isolation, spectral analysis |
| **Conservation laws** | All five known K(m,m) conservation laws (I₀–I₄) |
| **Shelf waves** | Profile computation for the compacton singular limit (Theorem 7.6) |
| **Symbolic verification** | SymPy-based verification of Theorem 4.1 and Proposition 5.3 (notebook) and of the Appendix C identities ([`verification/symbolic_verification.py`](verification/symbolic_verification.py)) |
| **Notebooks** | Colab-ready notebooks: symbolic verification, BBM collisions, shelf waves, and a finite-difference cross-check of the collisions |
| **Figure data** | CSV files used by pgfplots in the LaTeX manuscript |
| **DCDS-A companion** | Index of every file the paper names, plus the §5.1 and §6 scripts: [papers/hyman2026structure/](papers/hyman2026structure/README.md) |

---

## Quick Start

### Option 1: Google Colab (no installation)

Click any notebook link below to run directly in your browser:

| Notebook | Description |
|----------|-------------|
| [**Multi-m Debris Spectra**](https://colab.research.google.com/github/machyman/kmm-compactons/blob/main/notebooks/production_multi_m_D3v4.ipynb) | Finite-difference cross-check of the K(m,m) collision debris analysis, m = 2,3,4,5 (N = 1024, q = 3, δ = 0.5); not the PSM run of Figure 4, and its values differ |
| [**BBM Collisions**](https://colab.research.google.com/github/machyman/kmm-compactons/blob/main/notebooks/bbm_collision_analysis.ipynb) | Section 5.2: BBM comparison experiment |
| [**Shelf Wave Profiles**](https://colab.research.google.com/github/machyman/kmm-compactons/blob/main/notebooks/shelf_wave_figures.ipynb) | Section 7: Compacton singular limit |
| [**Symbolic Verification**](https://colab.research.google.com/github/machyman/kmm-compactons/blob/main/notebooks/symbolic_verification.ipynb) | VEE: Independent verification of key formulas |

### Option 2: Local Installation

```bash
git clone https://github.com/machyman/kmm-compactons.git
cd kmm-compactons
pip install -r requirements.txt

# Run the verification suite
python verification/verify_paper_results.py
python verification/symbolic_verification.py   # Appendix C identities (SymPy)

# Run tests
pytest tests/ -v
```

**Requirements:** Python 3.10 or 3.11, NumPy ≥ 1.24, SciPy ≥ 1.11, Matplotlib ≥ 3.7.  
Optional: SymPy ≥ 1.12 (symbolic verification).

---

## Repository Structure

```
kmm-compactons/
├── src/                        # Python source modules
│   ├── __init__.py
│   ├── compacton_physics.py    # Universal formulas (Theorem 4.1)
│   ├── kmm_solver.py           # K(m,m) FD solver with spectral filtering
│   ├── bbm_solver.py           # BBM pseudospectral solver
│   ├── conservation.py         # Five conservation laws I₀–I₄
│   ├── collision_analysis.py   # Debris extraction and spectral analysis
│   ├── shelf_waves.py          # Shelf wave ODE solver
│   ├── shelf_wave_profiles.py  # Exact K(2,2) shelf-wave profiles by quadrature
│   └── plotting.py             # Publication figure utilities
├── notebooks/                  # Jupyter notebooks (Colab-ready)
│   ├── production_multi_m_D3v4.ipynb   # FD cross-check of the collisions (not Figure 4)
│   ├── bbm_collision_analysis.ipynb    # BBM comparison (Section 5.2)
│   ├── shelf_wave_figures.ipynb        # Shelf waves (Section 7)
│   └── symbolic_verification.ipynb     # Formula verification (VEE)
├── data/                       # Computational output data
│   ├── measurements_D3v4_N1024.json    # K(m,m) collision measurements
│   └── BBM_Collision_Results.json      # BBM collision measurements
├── paper/                      # Figure sources for the manuscript
│   ├── fig_*.tex                      # pgfplots/TikZ sources, Figures 1, 2, 6 and 8
│   ├── make_figures_v2_3_0.py         # Figures 3 and 7, Supplement Figure S1
│   ├── run_spacetime_collision.py     # regenerates spacetime_k22.npz (Figure 3 data)
│   └── spacetime_k22.npz              # Figure 3 space-time data
├── paper/figdata/              # CSV data for LaTeX pgfplots figures
│   ├── profile_compacton.csv           # Compacton profile (a=0 limit)
│   ├── profile_a0.01.csv              # Shelf wave, a=0.01
│   ├── profile_a0.10.csv              # Shelf wave, a=0.10
│   ├── profile_a0.50.csv              # Shelf wave, a=0.50
│   ├── profile_a1.00.csv              # Shelf wave, a=1.00
│   ├── period_energy.csv              # Period-energy relationship
│   └── regime_*.csv                   # Figure 6 profiles: compacton, solitary, near-critical, cnoidal
├── papers/hyman2026structure/  # DCDS-A companion: index, §5.1 and §6 scripts
├── tests/                      # Pytest test suite
│   └── test_compacton.py
├── verification/               # Paper result verification
│   ├── verify_paper_results.py
│   └── symbolic_verification.py   # Appendix C identities (SymPy)
├── requirements.txt
├── pyproject.toml
├── CITATION.cff
├── LICENSE                     # MIT
└── README.md                   # This file
```

---

## Numerical Methods

### K(m,m) Solvers

Every K(m,m) collision run in the paper used the pseudospectral method (PSM) of Appendix D. The PSM package in `papers/hyman2026structure/src/kmm_compactons/` is the solver behind §5.1 and Figure 3; see [papers/hyman2026structure/](papers/hyman2026structure/README.md#the-pseudospectral-package).

The finite-difference solver (`src/kmm_solver.py`), the scheme of the cross-check notebook `production_multi_m_D3v4.ipynb`, uses a regularized finite-difference approach:

- **Spatial discretization:** 4th-order central differences for D₁ and D₃, 2nd-order for D₂
- **Spectral filtering:** Low-pass filter on advection, high-pass filter on artificial dissipation
- **Time integration:** Classical RK4 with CFL-based timestep
- **Boundary conditions:** Periodic

The key innovation is dual spectral filtering that suppresses aliasing while preserving physical modes below the cutoff wavenumber κ_cut = 2πN/(2qL).  This produces clean debris spectra enabling the spectral notch analysis at k = 1 (evidence for Theorem 5.5(c)).

### BBM Solver

The BBM comparison solver (`src/bbm_solver.py`) uses:

- **Pseudospectral discretization** with 2/3-rule dealiasing
- **Implicit time stepping** via the Fourier symbol (1 + k²)⁻¹
- **Adaptive RK45** integration (SciPy `solve_ivp`)

---

## Key Results Reproduced

| Paper Reference | Result | Notebook/Script |
|-----------------|--------|-----------------|
| Theorem 4.1 | Universal formulas α, ℓ, c(A) | `symbolic_verification.ipynb` |
| Proposition 5.3 | I₂ = I₃ = 0 for compactons | `symbolic_verification.ipynb` |
| Table 4, Table S2 | Finite-difference cross-check of the collision measurements, m = 2,3,4,5 (N = 1024, q = 3, δ = 0.5); not the PSM run of the tables, and its values differ | `production_multi_m_D3v4.ipynb` |
| Figure 4 | Multi-m debris spectra (PSM run): generator not yet in this repository. The notebook plots the finite-difference cross-check (N = 1024, q = 3, δ = 0.5), whose values differ | `production_multi_m_D3v4.ipynb` (cross-check only) |
| Tables S3–S5 | BBM collision data, in [`data/BBM_Collision_Results.json`](data/BBM_Collision_Results.json); the notebook re-runs the Table S3 cases (r = 0.5), and its values differ | `bbm_collision_analysis.ipynb` |
| Figure 8 | Shelf wave profiles: `src/shelf_wave_profiles.py` reproduces the deposited `paper/figdata/profile_*.csv` to within 6.35 × 10⁻⁴; `shelf_wave_figures.ipynb` writes CSVs on other grids and does not produce those files | [`src/shelf_wave_profiles.py`](src/shelf_wave_profiles.py) |
| Section 5.1 | Collision amplitudes and elasticity | [`collision_certification.py`](papers/hyman2026structure/verification/collision_certification.py) |
| Table 5 (Section 6) | Compacton emergence | [`emergence_verification.py`](papers/hyman2026structure/verification/emergence_verification.py) |
| Table 8 (Appendix C) | Section 7 identities | [`symbolic_verification.py`](verification/symbolic_verification.py) |

---

## Correspondence: Paper ↔ Code

| Paper Symbol | Code Variable | Module |
|-------------|---------------|--------|
| α = 2/(m−1) | `edge_exponent(m)` | `compacton_physics` |
| ℓ = 2m/(m−1) | `width_parameter(m)` | `compacton_physics` |
| c = (m+1)/(2m) A^{m−1} | `speed(m, A)` | `compacton_physics` |
| I₀ = ∫ u dx | `mass(u, dx)` | `conservation` |
| I₁ = ∫ u^{m+1}/(m+1) dx | `energy(u, m, dx)`, which returns ∫ u^{m+1} dx = (m+1) I₁ | `conservation` |
| I₄ (nonlocal) | `nonlocal_momentum(u, x, dx)` | `conservation` |
| ε (elasticity) | `elasticity_coefficient(...)` | `collision_analysis` |
| κ (decay rate) | `decay_rate_theory(m, a, c)` | `shelf_waves` |

---

## Citation

If you use this code in your research, please cite both the paper and the software:

```bibtex
@article{hyman2026structure,
  author  = {Hyman, James M.},
  title   = {What Remains When Integrability Fails? {M}athematical
             Structure in the {K}(m,m) Compacton Equations},
  journal = {Discrete and Continuous Dynamical Systems -- Series A},
  year    = {2026},
  note    = {Peter D. Lax Centennial Memorial Volume},
}
```

See also `CITATION.cff` for machine-readable citation metadata.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Author

**James M. Hyman**
Department of Mathematics, Tulane University
mhyman@tulane.edu
ORCID: [0000-0001-5247-5794](https://orcid.org/0000-0001-5247-5794)
