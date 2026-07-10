"""Quick look at a sparse point cloud: 3 projections colored by RGB, plus stats.
Usage: python viz_pointcloud.py sparse.ply out.png"""
import sys, numpy as np, open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ply, out = sys.argv[1], sys.argv[2]
pcd = o3d.io.read_point_cloud(ply)
P = np.asarray(pcd.points); C = np.asarray(pcd.colors)
if C.size == 0:
    C = None
print("points:", len(P))

# drop far outliers for a clean view (keep inner 92% around the median)
med = np.median(P, axis=0)
d = np.linalg.norm(P - med, axis=1)
keep = d < np.percentile(d, 92)
P = P[keep]; C = C[keep] if C is not None else None
print("after outlier trim:", len(P))
print("bbox extent (arbitrary units):", np.round(P.max(0) - P.min(0), 2))

pairs = [(0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z")]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for k, (a, b, name) in enumerate(pairs):
    ax[k].scatter(P[:, a], P[:, b], s=2, c=(C if C is not None else "k"), linewidths=0)
    ax[k].set_title(f"projection {name}"); ax[k].set_aspect("equal", "box")
    ax[k].set_xlabel(name[0]); ax[k].set_ylabel(name[2])
fig.suptitle(f"A74 sparse reconstruction — {len(P)} pts (3 orthographic views)")
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
