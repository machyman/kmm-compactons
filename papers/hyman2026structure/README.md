# hyman2026structure: companion files for the DCDS-A paper

James M. Hyman, "What Remains When Integrability Fails? Mathematical Structure in the K(m,m)
Compacton Equations," *Discrete and Continuous Dynamical Systems – Series A*, Peter D. Lax Centennial
Memorial Volume (2026).

This page lists every repository file that the paper and its Supplement name, and the scripts that
reproduce specific results. The repository's main page is [README.md](../../README.md).

## Files named in the Supplement (§S9)

| Printed name | Location | Description in the Supplement |
|---|---|---|
| `kmm_solver.py` | [src/kmm_solver.py](../../src/kmm_solver.py) | K(m,m) regularized finite-difference solver with dual spectral filtering |
| `bbm_solver.py` | [src/bbm_solver.py](../../src/bbm_solver.py) | BBM pseudospectral solver |
| `collision_analysis.py` | [src/collision_analysis.py](../../src/collision_analysis.py) | Post-collision amplitude extraction and debris spectral analysis |
| `shelf_wave_profiles.py` | [src/shelf_wave_profiles.py](../../src/shelf_wave_profiles.py) | Exact shelf-wave profile computation from quadrature formulas |
| `symbolic_verification.py` | [verification/symbolic_verification.py](../../verification/symbolic_verification.py) | SymPy scripts for Appendix C verification |
| `production_multi_m_D3v4.ipynb` | [notebooks/production_multi_m_D3v4.ipynb](../../notebooks/production_multi_m_D3v4.ipynb) | Colab-compatible notebook reproducing all collision results |
| `tests/` | [tests/test_compacton.py](../../tests/test_compacton.py) | Validation test suite |
| `measurements_D3v4_N1024.json` | [data/measurements_D3v4_N1024.json](../../data/measurements_D3v4_N1024.json) | Complete K(m,m) collision dataset |
| `BBM_Collision_Results.json` | [data/BBM_Collision_Results.json](../../data/BBM_Collision_Results.json) | Complete BBM collision dataset |

## Reproducing specific results

| Result in the paper | Script | Runtime | Agreement |
|---|---|---|---|
| §5.1 collision amplitudes, elasticity, debris | [verification/collision_certification.py](verification/collision_certification.py) | under a minute | every printed digit |
| §6 emergence table (Table 5) | [verification/emergence_verification.py](verification/emergence_verification.py) | under a minute | see [EMERGENCE_COMPARISON.md](EMERGENCE_COMPARISON.md) |
| §6 sensitivity to unstated settings | [verification/emergence_sensitivity.py](verification/emergence_sensitivity.py) | a few minutes | see [EMERGENCE_COMPARISON.md](EMERGENCE_COMPARISON.md) |
| Appendix C, Table 8, and Proposition 7.3 | [symbolic_verification.py](../../verification/symbolic_verification.py) | under a minute | all 70 checks pass |
| Figure 7 and Supplement Figure S2 data | [shelf_wave_profiles.py](../../src/shelf_wave_profiles.py) | seconds | see "Figure data" below |

The §6 table came from a production run whose full settings are not recorded here;
[EMERGENCE_COMPARISON.md](EMERGENCE_COMPARISON.md) documents what the published script reproduces.

## Figure data

The ten CSV files in `paper/figdata/` are the files the manuscript's figures read, byte-identical to those
in the submitted package.

- [`profile_compacton.csv`](../../paper/figdata/profile_compacton.csv) and [`period_energy.csv`](../../paper/figdata/period_energy.csv) are reproduced to their printed precision by
  `shelf_wave_profiles.py` (exact compacton formula; tanh-sinh quadrature of the period integral).
- [`profile_a0.01.csv`](../../paper/figdata/profile_a0.01.csv), [`profile_a0.10.csv`](../../paper/figdata/profile_a0.10.csv),
  [`profile_a0.50.csv`](../../paper/figdata/profile_a0.50.csv) and [`profile_a1.00.csv`](../../paper/figdata/profile_a1.00.csv) came from a separate numerical integration whose settings
  are not recorded. They agree with the exact quadrature to within 7 × 10⁻⁴, about 0.01% of the crest
  height; running `shelf_wave_profiles.py` prints the comparison.
- [`regime_algebraic.csv`](../../paper/figdata/regime_algebraic.csv), [`regime_cnoidal.csv`](../../paper/figdata/regime_cnoidal.csv),
  [`regime_compacton.csv`](../../paper/figdata/regime_compacton.csv) and [`regime_solitary.csv`](../../paper/figdata/regime_solitary.csv) (Figure 6): their generator is not yet in this repository.

## The pseudospectral package

`src/kmm_compactons/` holds the pseudospectral K(m,n) solver ([`solver.py`](src/kmm_compactons/solver.py), class
`KMNSolver`) and its supporting modules, used by the §5.1 and §6 scripts. Its tests are
[`test_compacton.py`](tests/test_compacton.py), [`test_conservation.py`](tests/test_conservation.py) and
[`test_shelf_waves.py`](tests/test_shelf_waves.py); its packaging is [`pyproject.toml`](pyproject.toml). It needs Python 3.11 and NumPy 2 or later.
The scripts put the package on their own import path, so they run without installation:

```bash
python papers/hyman2026structure/verification/collision_certification.py
PYTHONPATH=papers/hyman2026structure/src pytest papers/hyman2026structure/tests
```

Changes from the package as archived on 2026-05-20: `solver.py` imports `compacton_speed` from
`compacton_profiles` (the archived copy imported it from `compacton`, which does not define it);
module docstrings cite the paper's final numbering; the distribution is named
`kmm-compactons-hyman2026structure` so that installing it does not replace the repository's root package.
