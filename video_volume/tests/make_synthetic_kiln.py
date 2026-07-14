"""Generate a synthetic Kon-Tiki kiln point cloud filled with biochar to a KNOWN
height (and optional heap), for validating estimate_volume.py against ground truth.

The cloud contains: the frustum wall, the biochar top surface, and a wide ground
plane — then an arbitrary rotation/translation/scale, to mimic a reconstruction
whose frame and scale are unknown (exactly what estimate_volume must recover)."""
import numpy as np

R, RB, H = 75.0, 41.15, 93.0


def true_volume_L(h_fill, heap_cm=0.0):
    """Exact biochar volume: frustum fill + parabolic heap dome."""
    h = float(np.clip(h_fill, 0, H))
    rs = RB + (R - RB) * h / H
    Vf = (1 / 3) * np.pi * h * (RB ** 2 + RB * rs + rs ** 2)
    Vh = heap_cm * np.pi * rs ** 2 / 2.0          # integral of heap*(1-(r/rs)^2)
    return (Vf + Vh) / 1000.0


def _rand_rot(rng):
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(c), -np.sin(c)], [0, np.sin(c), np.cos(c)]])
    return Rz @ Ry @ Rx


def make(h_fill, n_wall=30000, n_surf=25000, n_ground=25000, heap_cm=0.0,
         noise=0.3, transform=True, seed=0, arc_deg=360.0, interior_n=0):
    rng = np.random.default_rng(seed)
    arc = np.radians(arc_deg)                          # <360 = partial orbit
    # frustum wall, radius r(z) = RB + (R-RB) z/H, over full depth
    z = rng.uniform(0, H, n_wall); r = RB + (R - RB) * z / H; th = rng.uniform(0, arc, n_wall)
    wall = np.c_[r * np.cos(th), r * np.sin(th), z]
    # biochar top surface at z=h_fill, radius r_s, with a parabolic heap
    rs = RB + (R - RB) * h_fill / H
    u = np.sqrt(rng.uniform(0, 1, n_surf)); rr = rs * u; ph = rng.uniform(0, arc, n_surf)
    zz = h_fill + heap_cm * (1 - u ** 2)
    surf = np.c_[rr * np.cos(ph), rr * np.sin(ph), zz]
    # wide ground plane at z=0 outside the kiln footprint
    gx = rng.uniform(-1.5 * R, 1.5 * R, n_ground); gy = rng.uniform(-1.5 * R, 1.5 * R, n_ground)
    keep = np.hypot(gx, gy) > R * 1.03
    ground = np.c_[gx[keep], gy[keep], np.zeros(keep.sum())]
    parts = [wall, surf, ground]
    if interior_n:                                     # deep interior artefacts (VGGT-like)
        zi = rng.uniform(H * 0.1, max(h_fill, H * 0.15), interior_n)
        rmax = RB + (R - RB) * zi / H
        ri = rmax * np.sqrt(rng.uniform(0, 1, interior_n)); ti = rng.uniform(0, arc, interior_n)
        parts.append(np.c_[ri * np.cos(ti), ri * np.sin(ti), zi])
    P = np.vstack(parts).astype(float)
    P += rng.normal(0, noise, P.shape)
    if transform:
        P = (P @ _rand_rot(rng).T) * rng.uniform(0.3, 3.0) + rng.uniform(-40, 40, 3)
    return P
