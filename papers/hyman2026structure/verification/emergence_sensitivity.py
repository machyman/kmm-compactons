"""
emergence_sensitivity.py: sensitivity of the Section 6 emergence rows to solver settings the
paper does not state. Companion to emergence_verification.py; the results are summarized in
../EMERGENCE_COMPARISON.md ("Sensitivity to settings the paper does not state").

1. delta = 0.15, 0.20, 0.25, 0.30 for K(3,3) and K(3,2).
2. Eight solver configurations at delta = 0.20 (filter exponential or sharp, dissipation
   power 4 or 2, spatial ramp on or off) for K(2,2), K(3,3), K(4,4) and K(3,2).
3. K(3,2) started from a K(2,2)-shaped pulse instead of a K(3,3)-shaped one.

Every peak at or above the 0.08 floor is printed unrounded.
Run from anywhere:  python papers/hyman2026structure/verification/emergence_sensitivity.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from emergence_verification import TABLE, run   # noqa: E402


def fmt(pk):
    return "[" + ", ".join(f"{a:.3f}" for a, _ in pk) + "]"


def main():
    print("1. delta sweep")
    for (m, n) in [(3, 3), (3, 2)]:
        for d in (0.15, 0.20, 0.25, 0.30):
            pk, *_ = run(m, n, delta=d)
            print(f"   K({m},{n}) delta = {d:.2f}: N = {len(pk)} {fmt(pk)}", flush=True)
    print("2. configuration sweep at delta = 0.20")
    for fs, dp, ramp in itertools.product(['exponential', 'sharp'], [4, 2], [True, False]):
        rows = []
        for (m, n) in TABLE:
            try:
                pk, *_ = run(m, n, filter_shape=fs, diss_power=dp, hp_ramp=ramp)
                rows.append(f"K({m},{n}) {fmt(pk)}")
            except Exception as e:           # report and continue, as a sweep should
                rows.append(f"K({m},{n}) ERROR {type(e).__name__}")
        print(f"   [{fs:11s} p={dp} ramp={str(ramp):5s}] " + " | ".join(rows), flush=True)
    print("3. K(3,2) from a K(2,2)-shaped pulse")
    pk, *_ = run(3, 2, mshape=2)
    print(f"   K(3,2), mshape = 2: N = {len(pk)} {fmt(pk)}", flush=True)


if __name__ == "__main__":
    main()
