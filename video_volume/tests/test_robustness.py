"""Adversarial robustness test for estimate_volume.py.

Confirms the engine DOES NOT return a confident-but-wrong number on bad input:
garbage / empty / partial-orbit / wrong-shape clouds must be flagged, and a good
full-orbit kiln must read 'good'. It must never crash or return NaN."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from estimate_volume import estimate_points
from make_synthetic_kiln import make

fails = 0
def check(name, ok, info=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {info}")
    if not ok: fails += 1

# 1) good full-orbit kiln -> confidence 'good', no crash, finite
r = estimate_points(make(70, seed=3))
check("good full kiln -> 'good'", r["confidence"] == "good",
      f"(conf={r['confidence']}, cover={r['angular_coverage']:.2f}, vol={r['volume_L']:.0f})")
check("volume is finite", np.isfinite(r["volume_L"]))

# 2) partial orbit (~170 deg) -> must flag incomplete orbit, not 'good'
r = estimate_points(make(70, seed=3, arc_deg=170))
check("partial orbit flagged", any("incomplete orbit" in w for w in r["warnings"]),
      f"(conf={r['confidence']}, cover={r['angular_coverage']:.2f})")

# 3) random garbage cloud -> must NOT be 'good', must not crash/NaN
g = np.random.default_rng(1).normal(scale=50, size=(4000, 3))
r = estimate_points(g)
check("garbage not trusted", r["confidence"] != "good", f"(conf={r['confidence']})")
check("garbage volume finite", np.isfinite(r["volume_L"]))

# 4) tiny/empty input -> unreliable, no crash
r = estimate_points(np.zeros((40, 3)))
check("tiny input -> unreliable", r["confidence"] == "unreliable")

# 5) wrong shape: a straight cylinder (not a cone) -> flag shape mismatch
rng = np.random.default_rng(2); th = rng.uniform(0, 2*np.pi, 40000); zc = rng.uniform(0, 90, 40000)
cyl = np.c_[70*np.cos(th), 70*np.sin(th), zc]
gx = rng.uniform(-110, 110, 20000); gy = rng.uniform(-110, 110, 20000); keep = np.hypot(gx, gy) > 80
cyl = np.vstack([cyl, np.c_[gx[keep], gy[keep], np.zeros(keep.sum())]])
r = estimate_points(cyl)
check("wrong-shape (cylinder) flagged", r["confidence"] != "good" and
      any("shape" in w or "orbit" in w or "scale" in w for w in r["warnings"]),
      f"(conf={r['confidence']}, warns={len(r['warnings'])})")

print("\nROBUSTNESS:", "PASS" if fails == 0 else f"{fails} FAILED")
