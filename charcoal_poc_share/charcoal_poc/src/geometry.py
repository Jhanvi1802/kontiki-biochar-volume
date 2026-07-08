#!/usr/bin/env python
"""
Stage 2-4 of the shared pipeline: fill mask -> fill height -> volume -> weight.

For the synthetic track we know the container is a vertical cylinder of radius r
with its base on the world plane z=0 and axis = world z-axis, and we have the
exact camera calibration (K, R, t in OpenCV world->cam convention). That lets us
recover the real-world fill height by back-projecting the TOP boundary of the
charcoal mask and intersecting each ray with the cylinder wall x^2+y^2=r^2.

The raw height carries a small, systematic bias (the granular heap and viewing
parallax), so we fit a 2-parameter linear calibration h_est = a*h_raw + b on the
TRAIN split and apply it on the TEST split -- exactly the "calibrate heap/parallax
offset on synthetic ground truth" step in the plan. No per-image GT is used at
test time.
"""
import numpy as np

# ----------------------------------------------------------------- camera math
def cam_center_and_dirs(K, R, t, uv):
    """Back-project pixels uv (N,2) to world rays. Returns (C, D) with C the
    camera centre (3,) and D unit ray directions (N,3). OpenCV convention."""
    K = np.asarray(K, float); R = np.asarray(R, float); t = np.asarray(t, float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    uv = np.asarray(uv, float)
    d_cam = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy,
                      np.ones(len(uv))], axis=1)               # (N,3)
    Rt = R.T
    C = -Rt @ t
    D = d_cam @ Rt.T                                            # rotate to world
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    return C, D

def ray_cylinder_z(C, D, r, zmax, far=True):
    """Intersect rays X=C+sD with infinite cylinder x^2+y^2=r^2. Return world z
    of the chosen hit (far=larger s, i.e. the back wall) within [0, zmax], else nan."""
    Cx, Cy = C[0], C[1]
    Dx, Dy = D[:, 0], D[:, 1]
    a = Dx**2 + Dy**2
    b = 2 * (Cx * Dx + Cy * Dy)
    c = Cx**2 + Cy**2 - r**2
    disc = b**2 - 4 * a * c
    z = np.full(len(D), np.nan)
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.clip(disc, 0, None))
    s1 = (-b - sq) / (2 * a)
    s2 = (-b + sq) / (2 * a)
    s = np.where(far, np.maximum(s1, s2), np.minimum(s1, s2))
    zc = C[2] + s * D[:, 2]
    good = ok & (s > 0) & (zc >= -1e-3) & (zc <= zmax + 1e-3)
    z[good] = zc[good]
    return z

# ----------------------------------------------------------------- fill height
def axis_image_u(K, R, t, zmax):
    """Image column of the container central axis (project axis midpoint)."""
    K = np.asarray(K, float)
    X = np.array([0, 0, zmax * 0.5])
    xc = np.asarray(R, float) @ X + np.asarray(t, float)
    return K[0, 0] * xc[0] / xc[2] + K[0, 2]

def estimate_fill_height_raw(fill_mask, K, R, t, r, zmax, band_frac=0.5):
    """Robust raw fill height (m) from the binary charcoal mask + calibration.

    For columns in a central band around the container axis, take the topmost
    charcoal pixel (smallest v = highest in image), back-project, intersect the
    cylinder wall and read world z. Median over the band = back-rim height ~ h."""
    ys, xs = np.where(fill_mask > 0)
    if len(xs) < 30:
        return np.nan
    u_axis = axis_image_u(K, R, t, zmax)
    half = band_frac * 0.5 * (xs.max() - xs.min() + 1)
    cols = np.unique(xs[(xs >= u_axis - half) & (xs <= u_axis + half)])
    if len(cols) < 5:
        cols = np.unique(xs)
    H_img = fill_mask.shape[0]
    run = 6                            # require a vertical run of charcoal below
    uv_top = []
    for u in cols:
        col = fill_mask[:, u] > 0
        vv = np.where(col)[0]
        if len(vv) == 0:
            continue
        # topmost row whose next `run` pixels are mostly charcoal (reject specks)
        v = None
        for cand in vv:
            lo, hi = cand, min(cand + run, H_img)
            if col[lo:hi].mean() >= 0.6:
                v = cand; break
        if v is None:
            v = vv.min()
        uv_top.append((u, v))
    if len(uv_top) < 3:
        return np.nan
    uv_top = np.array(uv_top, float)
    C, D = cam_center_and_dirs(K, R, t, uv_top)
    z = ray_cylinder_z(C, D, r, zmax, far=True)
    z = z[np.isfinite(z)]
    if len(z) < 3:
        return np.nan
    return float(np.median(z))

# ----------------------------------------------------------------- volume/weight
def volume_liters(r, h, theta_deg=None):
    """Cylinder body (+ optional heaped cone) volume in litres."""
    V = np.pi * r * r * h
    if theta_deg is not None:
        h_cone = r * np.tan(np.radians(theta_deg))
        V += (1.0 / 3.0) * np.pi * r * r * h_cone
    return V * 1000.0

def weight_kg(volume_liters_val, density_kgm3):
    return density_kgm3 * (volume_liters_val / 1000.0)

# ----------------------------------------------------------------- calibration
def fit_linear(h_raw, h_true):
    """Least-squares a,b for h_true ~ a*h_raw + b (ignores nans)."""
    h_raw = np.asarray(h_raw, float); h_true = np.asarray(h_true, float)
    m = np.isfinite(h_raw) & np.isfinite(h_true)
    A = np.stack([h_raw[m], np.ones(m.sum())], axis=1)
    coef, *_ = np.linalg.lstsq(A, h_true[m], rcond=None)
    return float(coef[0]), float(coef[1])

def apply_linear(h_raw, ab):
    return ab[0] * np.asarray(h_raw, float) + ab[1]
