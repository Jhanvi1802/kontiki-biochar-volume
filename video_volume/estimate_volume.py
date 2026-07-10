"""Video -> Volume pipeline, VOLUME STEP (local, CPU — no GPU needed).

Takes a 3-D point cloud of a biochar-filled Kon-Tiki kiln (from the reconstruction
step) and returns the biochar volume in litres. Pure geometry:
  1. find 'up' from the dominant plane; put the *widest* horizontal sheet (ground)
     at the bottom  (so we never confuse the biochar surface for the ground),
  2. isolate the kiln, fit the rim -> scale the cloud to real cm (rim radius 75 cm),
  3. integrate the measured biochar surface against the known kiln cone, filling
     gaps by nearest-neighbour so sparse spots don't under-count.

Same logic the Colab notebook uses; it runs here on CPU because it is not GPU work.

Usage:  python estimate_volume.py cloud.ply [rim_radius_cm] [views.png]
"""
import sys, numpy as np
from scipy.interpolate import NearestNDInterpolator
from scipy.spatial import cKDTree

# Kon-Tiki 1000 geometry (cm), from the design drawing
R_CM, RB_CM, H_CM = 75.0, 41.15, 93.0   # rim Ø1500, bottom Ø823, depth 930 (design drawing)
DENSITY = 0.25  # kg / L
CELL = 3.0      # integration grid (cm)
TOP_PCT = 12    # per-cell percentile = the top (biochar) surface, robust to deep artefacts
COL_MIN = 9.0   # cm: min biochar column to count (rejects the steep-wall ring; ~CELL*H/(R-RB))


def rot_from_to(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def fit_circle(xy):
    x, y = xy[:, 0], xy[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]; b = x ** 2 + y ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = c[0], c[1]
    return cx, cy, np.sqrt(max(c[2] + cx ** 2 + cy ** 2, 1e-9))


def _segment_plane(P, thr, iters=2000, seed=0):
    """Minimal RANSAC plane fit -> (normal, inlier_mask). No open3d dependency."""
    rng = np.random.default_rng(seed)
    best_n, best_in = None, None
    n_best = 0
    for _ in range(iters):
        idx = rng.choice(len(P), 3, replace=False)
        p0, p1, p2 = P[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        nl = np.linalg.norm(nrm)
        if nl < 1e-9:
            continue
        nrm = nrm / nl
        d = np.abs((P - p0) @ nrm)
        inl = d < thr
        c = int(inl.sum())
        if c > n_best:
            n_best, best_n, best_in = c, nrm, inl
    return best_n, best_in


def _largest_cluster(P, eps, min_pts=20):
    """Grid-based connected-components clustering (fast, no open3d)."""
    keys = np.floor(P / eps).astype(np.int64)
    from collections import defaultdict
    cell = defaultdict(list)
    for i, k in enumerate(map(tuple, keys)):
        cell[k].append(i)
    seen, best = set(), []
    neigh = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    for start in cell:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            c = stack.pop(); comp.extend(cell[c])
            for d in neigh:
                nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if nb in cell and nb not in seen:
                    seen.add(nb); stack.append(nb)
        if len(comp) > len(best):
            best = comp
    return np.array(best) if len(best) >= min_pts else np.arange(len(P))


def _wall_depth(rr):
    """Depth (cm, below rim) of the kiln wall/floor at radius rr (vectorised)."""
    return np.where(rr <= RB_CM, H_CM, (R_CM - rr) / (R_CM - RB_CM) * H_CM)


def estimate_points(P, rim_radius_cm=R_CM, views_png=None, heatmap_png=None, debug=False):
    P = np.asarray(P, float)
    P = P[np.isfinite(P).all(1)]
    med = np.median(P, 0); d = np.linalg.norm(P - med, axis=1)
    P = P[d < np.percentile(d, 98)]
    diag = float(np.linalg.norm(P.max(0) - P.min(0)))
    # scale-free local point spacing (robust to a huge ground plane in the scene)
    rng = np.random.default_rng(0)
    sub = P[rng.choice(len(P), min(len(P), 4000), replace=False)]
    spacing = float(np.median(cKDTree(P).query(sub, k=2)[0][:, 1]))

    # 1) up direction from the dominant plane (ground or biochar surface -> same normal)
    n, _ = _segment_plane(P, thr=max(2.5 * spacing, 0.003 * diag))
    R1 = rot_from_to(n, np.array([0, 0, 1.0]))
    Q = P @ R1.T
    z = Q[:, 2]; zr = z.max() - z.min()

    # 2) widest horizontal slab = ground; ensure it sits at the bottom
    nb, best_w, ground_z = 30, -1, None
    edges = np.linspace(z.min(), z.max(), nb + 1)
    for i in range(nb):
        m = (z >= edges[i]) & (z < edges[i + 1])
        if m.sum() < max(50, 0.004 * len(z)):
            continue
        c = Q[m, :2].mean(0)
        w = np.percentile(np.hypot(Q[m, 0] - c[0], Q[m, 1] - c[1]), 85)
        if w > best_w:
            best_w, ground_z = w, 0.5 * (edges[i] + edges[i + 1])
    if ground_z is None:
        ground_z = z.min()
    elif ground_z > 0.5 * (z.min() + z.max()):
        R1 = np.diag([1.0, -1.0, -1.0]) @ R1          # flip 180 deg about X
        Q = P @ R1.T; z = Q[:, 2]; ground_z = -ground_z

    # 3) drop the ground sheet, keep the largest cluster (the kiln)
    kiln = Q[z > ground_z + max(3 * spacing, 0.02 * zr)]
    idx = _largest_cluster(kiln, eps=3.0 * spacing)
    K = kiln[idx]

    # 3b) refine the axis: the kiln is a surface of revolution, so its symmetry
    # axis is the smallest-variance PCA direction (robust vs a tilted plane fit).
    c0 = K.mean(0)
    _, _, vt = np.linalg.svd(K - c0, full_matrices=False)
    axis = vt[2]
    if axis @ np.array([0, 0, 1.0]) < 0:
        axis = -axis
    K = (K - c0) @ rot_from_to(axis, np.array([0, 0, 1.0])).T
    # rim (wide end) must be at +Z: radius should grow with height
    rho = np.hypot(K[:, 0], K[:, 1])
    if np.corrcoef(K[:, 2], rho)[0, 1] < 0:
        K[:, 2] *= -1.0
    if debug:
        print(f"  [debug] spacing={spacing:.3f} n_kiln={len(K)}/{len(kiln)} axis_z={axis[2]:.3f}")

    # 4) fit the kiln WALL cone -> rim radius & plane -> scale to cm (axis at origin).
    # The wall's max-radius-vs-height is a straight line; extrapolate to the top.
    zk = K[:, 2]
    rho = np.hypot(K[:, 0], K[:, 1])
    zb = np.linspace(zk.min(), zk.max(), 22)
    zz, rr = [], []
    for i in range(len(zb) - 1):
        m = (zk >= zb[i]) & (zk < zb[i + 1])
        if m.sum() < 20:
            continue
        zz.append(0.5 * (zb[i] + zb[i + 1])); rr.append(np.percentile(rho[m], 98))
    zz, rr = np.array(zz), np.array(rr)
    m_slope, c_int = np.linalg.lstsq(np.c_[zz, np.ones_like(zz)], rr, rcond=None)[0]
    z_rim = float(np.percentile(zk, 99.5))
    r_units = m_slope * z_rim + c_int
    s = rim_radius_cm / r_units
    if debug:
        print(f"  [debug] slope={m_slope:.3f} r_units={r_units:.3f} s={s:.4f}")
    K = (K - np.array([0.0, 0.0, z_rim])) * s

    # 5) TOP-surface heightmap over the rim disk, integrated against the known cone.
    #    Per cell take the SHALLOWEST points (the biochar top) -> ignores deep
    #    interior / reconstruction artefacts, and works for any fill level.
    from collections import defaultdict
    x, y, zc = K[:, 0], K[:, 1], K[:, 2]
    dep = -zc; rho = np.hypot(x, y)
    V_full = (1 / 3) * np.pi * H_CM * (RB_CM ** 2 + RB_CM * R_CM + R_CM ** 2) / 1000.0
    colgrid = None                    # biochar-depth heatmap (filled in below)

    ins = rho <= R_CM
    gx = np.floor((x[ins] + R_CM) / CELL).astype(int)
    gy = np.floor((y[ins] + R_CM) / CELL).astype(int)
    depi = dep[ins]
    acc = defaultdict(list)
    for xi, yi, dp in zip(gx, gy, depi):
        acc[(xi, yi)].append(dp)
    cells, depths = [], []
    for key, v in acc.items():
        if len(v) < 3:
            continue
        cells.append(key); depths.append(np.percentile(v, TOP_PCT))   # top = biochar surface
    if len(cells) < 30:                               # essentially empty kiln
        V_L = V_simple = h_fill = 0.0
    else:
        cells = np.array(cells); depths = np.array(depths)
        centers = (cells + 0.5) * CELL - R_CM
        interp = NearestNDInterpolator(centers, depths)
        ncell = int(np.ceil(2 * R_CM / CELL))
        cc = (np.arange(ncell) + 0.5) * CELL - R_CM
        XX, YY = np.meshgrid(cc, cc); RR = np.hypot(XX, YY)
        disk = RR <= R_CM
        dsurf = interp(XX[disk], YY[disk])
        col = _wall_depth(RR[disk]) - dsurf
        V_L = float(col[col > COL_MIN].sum() * CELL * CELL / 1000.0)
        cg = _wall_depth(RR) - interp(XX, YY)          # full-grid biochar depth heatmap
        colgrid = np.where(disk & (cg > COL_MIN), cg, np.nan)
        # flat cross-check from the cells that actually hold biochar
        colc = _wall_depth(np.hypot(centers[:, 0], centers[:, 1])) - depths
        bio_d = depths[colc > COL_MIN]
        d_med = float(np.median(bio_d)) if len(bio_d) else float(np.median(depths))
        h_fill = float(np.clip(H_CM - d_med, 0, H_CM))
        rs = RB_CM + (R_CM - RB_CM) * (h_fill / H_CM)
        V_simple = (1 / 3) * np.pi * h_fill * (RB_CM ** 2 + RB_CM * rs + rs ** 2) / 1000.0
        if debug:
            p = np.percentile(depths, [10, 50, 90])
            print(f"  [debug] cells={len(depths)} top_depth p10/50/90="
                  f"{p[0]:.0f}/{p[1]:.0f}/{p[2]:.0f} V={V_L:.0f} Vflat={V_simple:.0f}")

    if views_png or heatmap_png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        depA = -K[:, 2]; rhoA = np.hypot(K[:, 0], K[:, 1])
        colA = _wall_depth(np.clip(rhoA, 0, R_CM)) - depA          # biochar beneath each pt
        is_bio = (rhoA <= R_CM) & (colA > COL_MIN)                 # biochar vs kiln structure

    if views_png:                                                  # 3-D reconstruction (2 panels)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(K[~is_bio, 0], K[~is_bio, 1], s=1, c="#cfc7b6", linewidths=0)
        if is_bio.any():
            ax[0].scatter(K[is_bio, 0], K[is_bio, 1], s=5, c=colA[is_bio], cmap="inferno", linewidths=0)
        ax[0].add_patch(Circle((0, 0), R_CM, fill=False, ec="#A94E28", lw=2))
        ax[0].set_title("TOP — biochar (colour) inside the kiln (grey)")
        ax[1].scatter(K[~is_bio, 0], K[~is_bio, 2], s=1, c="#cfc7b6", linewidths=0)
        if is_bio.any():
            ax[1].scatter(K[is_bio, 0], K[is_bio, 2], s=5, c=colA[is_bio], cmap="inferno", linewidths=0)
        ax[1].set_title("SIDE — biochar sits inside the Kon-Tiki cone")
        for a in ax:
            a.set_aspect("equal", "box")
        fig.tight_layout(); fig.savefig(views_png, dpi=130); plt.close(fig)

    if heatmap_png and colgrid is not None:                        # biochar heatmap on its own
        fig, ax = plt.subplots(figsize=(7.6, 6.4))
        im = ax.imshow(colgrid, origin="lower", extent=[-R_CM, R_CM, -R_CM, R_CM],
                       cmap="inferno", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="biochar depth (cm)")
        ax.add_patch(Circle((0, 0), R_CM, fill=False, ec="#A94E28", lw=1.8))
        ax.set_title("Biochar depth heatmap  ·  volume = sum of this")
        ax.set_aspect("equal", "box")
        fig.tight_layout(); fig.savefig(heatmap_png, dpi=130); plt.close(fig)

    return {"volume_L": V_L, "volume_L_flatfill": V_simple, "fill_height_cm": h_fill,
            "fill_pct": 100 * V_L / V_full, "weight_kg": DENSITY * V_L,
            "measured_rim_units": r_units, "scale_cm_per_unit": s}


def estimate(ply_path, rim_radius_cm=R_CM, views_png=None, heatmap_png=None):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(ply_path)
    if len(pcd.points) == 0:
        raise SystemExit("empty point cloud")
    return estimate_points(np.asarray(pcd.points), rim_radius_cm, views_png, heatmap_png)


def _print(res):
    print("=" * 46)
    print(f"  BIOCHAR VOLUME (integrated) : {res['volume_L']:6.0f} L")
    print(f"  cross-check (flat fill)     : {res['volume_L_flatfill']:6.0f} L")
    print(f"  fill height / fill %        : {res['fill_height_cm']:.0f} cm / {res['fill_pct']:.0f}%")
    print(f"  approx weight (~0.25 kg/L)  : {res['weight_kg']:6.0f} kg")
    print(f"  scale                       : {res['scale_cm_per_unit']:.4f} cm/unit")
    print("=" * 46)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ply = sys.argv[1]
    rim = float(sys.argv[2]) if len(sys.argv) > 2 else R_CM
    png = sys.argv[3] if len(sys.argv) > 3 else None
    _print(estimate(ply, rim, png))
