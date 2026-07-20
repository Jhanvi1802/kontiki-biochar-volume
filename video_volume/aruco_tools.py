"""ArUco marker tools for METRIC SCALE (the document's method).

The 3-D reconstruction has no real size on its own. We put printed ArUco markers of
a KNOWN size in the scene; this module (a) generates printable markers, and (b) reads
the true scale (cm per reconstruction-unit) from the markers using the per-pixel 3-D
points the reconstructor produces — so the size is measured from the video, not assumed.

Self-test:  python aruco_tools.py
"""
import os, numpy as np, cv2

DICT = "DICT_4X4_50"


def _dict(name=DICT):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def generate_markers(ids=(0, 1, 2, 3), out_dir="markers", px=700, border=80, name=DICT):
    """Save printable marker PNGs (white border kept). Print, MEASURE the black
    square's real edge, and pass that measured cm to scale_from_marker()."""
    os.makedirs(out_dir, exist_ok=True)
    d = _dict(name); paths = []
    for mid in ids:
        m = cv2.aruco.generateImageMarker(d, mid, px)
        canvas = np.full((px + 2 * border, px + 2 * border), 255, np.uint8)
        canvas[border:border + px, border:border + px] = m
        cv2.putText(canvas, f"{name} id={mid}", (border, border - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2, cv2.LINE_AA)
        p = os.path.join(out_dir, f"aruco_{mid}.png"); cv2.imwrite(p, canvas); paths.append(p)
    return paths


def scale_from_marker(images, world_points, marker_cm, name=DICT, debug=False):
    """images: list of frames at the SAME resolution as world_points (the images fed
    to the reconstructor). world_points: array [N, H, W, 3] of per-pixel 3-D positions.
    marker_cm: the MEASURED real edge length of the printed marker (cm).
    Returns dict(scale_cm_per_unit, n, spread_pct) or None if no marker seen."""
    d = _dict(name)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    world_points = np.asarray(world_points)
    scales = []
    for i in range(len(images)):
        img = images[i]
        gray = img if (img.ndim == 2) else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        wp = world_points[i]; H, W = wp.shape[:2]
        for c in corners:
            xy = np.round(c[0]).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, W - 1); xy[:, 1] = np.clip(xy[:, 1], 0, H - 1)
            P3 = wp[xy[:, 1], xy[:, 0]]                     # 4 corner 3-D positions
            if not np.isfinite(P3).all():
                continue
            edges = [np.linalg.norm(P3[(k + 1) % 4] - P3[k]) for k in range(4)]
            e = float(np.median(edges))
            if e > 1e-9:
                scales.append(marker_cm / e)
    if not scales:
        return None
    scales = np.array(scales); med = float(np.median(scales))
    res = dict(scale_cm_per_unit=med, n=len(scales),
               spread_pct=float(100 * np.std(scales) / max(med, 1e-9)))
    if debug:
        print("  [aruco]", res)
    return res


def _selftest():
    """Verify detection + the scale math on a synthetic flat plane of KNOWN scale."""
    px, brd = 240, 60
    m = cv2.aruco.generateImageMarker(_dict(), 0, px)
    img = np.full((px + 2 * brd, px + 2 * brd), 255, np.uint8)
    img[brd:brd + px, brd:brd + px] = m                    # marker spans px pixels
    Himg, Wimg = img.shape
    k = 0.5                                                 # units per pixel (synthetic)
    xs = np.arange(Wimg) * k; ys = np.arange(Himg) * k
    XX, YY = np.meshgrid(xs, ys)
    wp = np.stack([XX, YY, np.zeros_like(XX)], -1)[None]    # [1,H,W,3]
    marker_cm = 20.0                                        # pretend the real marker is 20 cm
    r = scale_from_marker([img], wp, marker_cm)
    # marker edge = px pixels = px*k units; true scale = 20 / (px*k)
    expected = marker_cm / (px * k)
    err = abs(r["scale_cm_per_unit"] - expected) / expected * 100
    print(f"detected n={r['n']}  scale={r['scale_cm_per_unit']:.4f}  expected={expected:.4f}  err={err:.2f}%")
    print("ARUCO SELF-TEST:", "PASS" if err < 2 else "FAIL")


if __name__ == "__main__":
    _selftest()
    print("generated:", generate_markers(out_dir=os.path.join(os.path.dirname(__file__), "markers")))
