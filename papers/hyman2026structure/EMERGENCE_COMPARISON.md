# §6 emergence table: what the published script reproduces

The emergence table in §6 of hyman2026structure (K(m,m) rows and the mixed K(3,2) row) came from
the author's production run of a mixed-exponent solver. That run's full settings are not recorded
in this repository. [`emergence_verification.py`](verification/emergence_verification.py) is an independent re-implementation that follows
every setting the paper states: wide-compacton pulse, width factor w = 2.5, A = 1, t = 175,
pseudospectral N_x = 512, q = 2, peak floor 0.08, δ = 0.20 (inside the stated range 0.15 to 0.30).
Settings the paper does not state were fixed as: domain L = 250, pulse centre x0 = 30, peak
separation 3.0, exponential filter, biharmonic dissipation (p = 4) with ramp. These solver settings
are the ones that reproduce every printed digit of the §5.1 collision values
([`collision_certification.py`](verification/collision_certification.py)).

## Result at that configuration

| row | table | script (4 decimals) | agreement at the table's precision |
|---|---|---|---|
| K(2,2) | 1.28, 0.72, 0.34, 0.11 | 1.2746, 0.7216, 0.3376, 0.1119 | last three; first 0.005 below |
| K(3,3) | 1.20, 0.79, 0.42, 0.10† | 1.1968, 0.7936, 0.4251, 0.1061 | first two; third sits on the rounding boundary; marginal fourth 0.106 vs 0.10† |
| K(4,4) | 1.16, 0.84, 0.50, none above 0.08 | 1.1546, 0.8378, 0.4931, 0.0838 | second; first 0.005 below, third 0.007 below; a trailing peak 0.084 just above the floor |
| K(3,2) | 1.33, 0.92, 0.47 | 1.3521, 0.9295, 0.4740 | count (three) and third; first two run 1.7% and 1.0% high |

## Sensitivity to settings the paper does not state

- δ from 0.15 to 0.30 moves every amplitude by less than 0.004; the largest shift is the K(3,2)
  first amplitude, 1.352 to 1.355. At δ = 0.15 a fourth K(3,2) peak appears just above the floor
  (0.084).
- Across eight configurations (filter exponential or sharp; dissipation p = 4 or 2; ramp on or
  off): K(2,2) is unchanged; K(3,2) first amplitude 1.342 to 1.352; K(4,4) third 0.492 to 0.498.
  Near-floor peaks move with these settings: without the ramp, K(3,3) gains a fifth peak at 0.101;
  with the sharp filter, the K(3,3) fourth peak rises to 0.116 to 0.117, a fifth appears at 0.107
  to 0.108, and K(3,2) gains one or two peaks between 0.082 and 0.097. The trailing K(4,4) peak
  stays between 0.082 and 0.086 in every configuration.
- A K(2,2)-shaped initial pulse for K(3,2) gives a first amplitude of 1.333 but shifts the others
  (0.914, 0.481) and adds a fourth peak at 0.084, so it is not the production setting either.

## Reading

The qualitative results reproduce: the compacton counts clearly above the floor, the ordering of
leading amplitudes (K(3,2) largest), and the K(m,m) amplitudes to within 0.007 of the table.
The mixed row runs up to 0.022 (1.7%) high. Entries within about 0.02 of the 0.08
floor depend on solver settings the paper does not list.

## Reproducing this page

[`verification/emergence_verification.py`](verification/emergence_verification.py) prints the four rows above, to four decimals, in under a minute.
[`verification/emergence_sensitivity.py`](verification/emergence_sensitivity.py) prints the δ sweep, the eight configurations and the K(2,2)-shaped pulse run behind the sensitivity notes, in a few minutes.
