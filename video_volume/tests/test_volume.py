"""Ground-truth test for estimate_volume.py.

Generates synthetic kilns filled to KNOWN volumes (varied fill, heap, orientation,
scale, seed) and checks the estimator recovers them. This validates the volume
MATH independently of reconstruction quality (which is the Colab/GPU step)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from estimate_volume import estimate_points
from make_synthetic_kiln import make, true_volume_L

CASES = [
    dict(h_fill=30, heap_cm=0,  seed=1),
    dict(h_fill=50, heap_cm=0,  seed=2),
    dict(h_fill=70, heap_cm=0,  seed=3),
    dict(h_fill=85, heap_cm=0,  seed=4),
    dict(h_fill=60, heap_cm=8,  seed=5),   # heaped surface
    dict(h_fill=45, heap_cm=12, seed=6),   # heaped surface
    dict(h_fill=75, heap_cm=0,  transform=False, seed=7),  # no transform
    dict(h_fill=80, heap_cm=5,  interior_n=15000, noise=0.6, seed=8),  # deep artefacts + noise
    dict(h_fill=55, heap_cm=0,  interior_n=10000, noise=0.6, seed=9),  # deep artefacts + noise
]

print(f"{'fill':>5}{'heap':>5}{'true_L':>9}{'est_L':>8}{'flat_L':>8}{'err%':>7}")
errs = []
for c in CASES:
    P = make(**c)
    r = estimate_points(P)
    true = true_volume_L(c['h_fill'], c.get('heap_cm', 0))
    err = 100 * (r['volume_L'] - true) / true
    errs.append(abs(err))
    print(f"{c['h_fill']:>5}{c.get('heap_cm',0):>5}{true:>9.0f}"
          f"{r['volume_L']:>8.0f}{r['volume_L_flatfill']:>8.0f}{err:>+7.1f}")

mae = np.mean(errs); mx = np.max(errs)
print(f"\nmean |err| = {mae:.1f}%   max |err| = {mx:.1f}%")
print("RESULT:", "PASS" if mx < 8 else "NEEDS WORK (max err >= 8%)")
