"""Video -> Volume pipeline, STEP 1: frame extraction.

Pulls sharp, evenly-spaced still frames from a slow kiln orbit video. These frames
feed the 3-D reconstruction step. Even spacing gives good overlap for matching;
the sharpness filter drops motion-blurred frames (the #1 killer of reconstruction).

Usage:
    python extract_frames.py "path/to/video.mp4" out_frames_dir [n_frames]
"""
import cv2, os, sys, numpy as np


def _sharpness(gray):
    """Variance of the Laplacian — higher = sharper (less motion blur)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract(video_path, out_dir, n_frames=36, candidates_per_slot=5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if total <= 0:  # some containers don't report a count — scan once to count
        total = 0
        while cap.grab():
            total += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"video: {w}x{h}  fps={fps:.1f}  frames={total}  dur={total/fps:.1f}s")
    n_frames = min(n_frames, max(1, total))
    slot = total / n_frames                      # frames per output slot

    saved, sharp_vals = [], []
    for i in range(n_frames):
        lo = int(i * slot); hi = int((i + 1) * slot)
        # sample a few candidate positions inside this slot, keep the sharpest
        cand_idx = np.linspace(lo, max(lo, hi - 1), candidates_per_slot).astype(int)
        best = None
        for idx in sorted(set(cand_idx.tolist())):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            s = _sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if best is None or s > best[1]:
                best = (idx, s, frame)
        if best is None:
            continue
        fn = os.path.join(out_dir, f"frame_{len(saved):03d}.jpg")
        cv2.imwrite(fn, best[2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(fn); sharp_vals.append(best[1])
    cap.release()

    sv = np.array(sharp_vals) if sharp_vals else np.array([0.0])
    print(f"saved {len(saved)} frames -> {out_dir}")
    print(f"sharpness  min={sv.min():.0f}  median={np.median(sv):.0f}  max={sv.max():.0f}")
    # crude quality read: <100 is very soft/blurry, low-res webshares often land here
    verdict = ("GOOD" if np.median(sv) > 300 else
               "MARGINAL" if np.median(sv) > 100 else "POOR (blurry / low-res)")
    print(f"frame quality for reconstruction: {verdict}")
    return saved


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    vid, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 36
    extract(vid, out, n)
