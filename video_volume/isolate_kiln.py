"""Try to isolate the kiln from a sparse cloud: remove the dominant (ground) plane,
cluster, keep the biggest compact cluster near the scene centre, and visualise it.
Usage: python isolate_kiln.py sparse.ply out.png"""
import sys, numpy as np, open3d as o3d
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ply, out = sys.argv[1], sys.argv[2]
pcd = o3d.io.read_point_cloud(ply)
P = np.asarray(pcd.points)
med = np.median(P, axis=0); d = np.linalg.norm(P - med, axis=1)
pcd = pcd.select_by_index(np.where(d < np.percentile(d, 92))[0])
P = np.asarray(pcd.points)
scale = np.linalg.norm(P.max(0) - P.min(0))
print("points:", len(P), " scene scale:", round(scale, 2))

# 1) remove the dominant plane (ground)
plane, inliers = pcd.segment_plane(distance_threshold=scale * 0.01,
                                   ransac_n=3, num_iterations=2000)
rest = pcd.select_by_index(inliers, invert=True)
print(f"ground plane removed: {len(inliers)} pts;  remaining: {len(rest.points)}")

# 2) cluster the rest
labels = np.array(rest.cluster_dbscan(eps=scale * 0.03, min_points=15))
R = np.asarray(rest.points)
if labels.max() < 0:
    print("no clusters found"); sys.exit(0)
# pick the cluster with the most points that sits near the scene centre
best, bestscore = None, -1
c0 = np.median(P, axis=0)
for lab in range(labels.max() + 1):
    idx = np.where(labels == lab)[0]
    pts = R[idx]
    center = pts.mean(0)
    score = len(idx) / (1 + np.linalg.norm(center - c0))
    if score > bestscore:
        bestscore, best = score, idx
K = R[best]
print(f"kiln candidate cluster: {len(K)} pts")
print("candidate extent:", np.round(K.max(0) - K.min(0), 2))

pairs = [(0, 1, "X-Y"), (0, 2, "X-Z"), (1, 2, "Y-Z")]
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for k, (a, b, name) in enumerate(pairs):
    ax[k].scatter(R[:, a], R[:, b], s=1, c="#cccccc", linewidths=0)
    ax[k].scatter(K[:, a], K[:, b], s=3, c="#A94E28", linewidths=0)
    ax[k].set_title(name); ax[k].set_aspect("equal", "box")
fig.suptitle(f"A74 kiln isolation — orange = kiln candidate ({len(K)} pts), grey = other")
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out)
