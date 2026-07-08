#!/usr/bin/env python
"""
CPU demo — runs the volume/weight pipeline LIVE on the bundled held-out test images.
No GPU, no training, no Blender. Needs only:  pip install numpy opencv-python-headless

What it does: for each test image it segments the charcoal (classical CV, no AI),
estimates the fill height by back-projecting onto the known cylinder, computes volume
& weight, and compares to the exact ground truth. It also runs the "perfect mask"
(GT) path to show the geometry ceiling.

  python demo_cpu.py            # classical + GT paths (pure numpy/opencv)
  python demo_cpu.py --yolo     # also run the trained AI (needs: pip install ultralytics)
"""
import os, json, argparse, cv2, numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import geometry as G, segment as S

HERE = os.path.dirname(os.path.abspath(__file__))
DD = os.path.join(HERE, "demo_data")
THETA = 38.0


def run(method, manifest, calib, yolo=None):
    ab = calib[method]
    errs_v, errs_w, ious = [], [], []
    for m in manifest:
        i = m["id"]
        rgb = cv2.imread(f"{DD}/rgb/{i:05d}.png")
        gt = cv2.imread(f"{DD}/fill_mask/{i:05d}.png", 0)
        if method == "gt":
            mask = gt
        elif method == "classical":
            mask = S.classical_fill_mask(rgb)
        elif method == "yolo":
            r = yolo.predict(rgb, verbose=False, retina_masks=True)[0]
            mask = np.zeros(gt.shape, np.uint8)
            if r.masks is not None:
                for mm in r.masks.data.cpu().numpy():
                    mask |= (cv2.resize(mm, (gt.shape[1], gt.shape[0])) > 0.5).astype(np.uint8) * 255
        ious.append(S.iou(mask, gt))
        cam = m["camera"]
        h = G.estimate_fill_height_raw(mask, cam["K"], cam["R"], cam["t"], m["radius_m"], m["height_m"])
        if not np.isfinite(h):
            continue
        hcal = ab[0] * h + ab[1]
        V = G.volume_liters(m["radius_m"], hcal, THETA)
        W = G.weight_kg(V, m["bulk_density_kgm3"])
        errs_v.append(abs(V - m["V_mesh_L"]) / m["V_mesh_L"] * 100)
        errs_w.append(abs(W - m["weight_kg"]) / m["weight_kg"] * 100)
    return np.mean(ious), np.mean(errs_v), np.mean(errs_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", action="store_true")
    ap.add_argument("--show", type=int, default=5, help="print this many per-image examples")
    args = ap.parse_args()
    manifest = json.load(open(f"{DD}/manifest.json"))
    calib = json.load(open(f"{DD}/calib.json"))
    print(f"Running the pipeline live on {len(manifest)} held-out test images (CPU)...\n")

    # a few per-image examples (classical)
    print("Examples (classical segmentation, no AI):")
    for m in manifest[:args.show]:
        i = m["id"]; rgb = cv2.imread(f"{DD}/rgb/{i:05d}.png")
        mask = S.classical_fill_mask(rgb); cam = m["camera"]
        h = G.estimate_fill_height_raw(mask, cam["K"], cam["R"], cam["t"], m["radius_m"], m["height_m"])
        ab = calib["classical"]
        V = G.volume_liters(m["radius_m"], ab[0]*h+ab[1], THETA) if np.isfinite(h) else float("nan")
        print(f"  img {i:05d}:  estimated {V:5.1f} L   |   true {m['V_mesh_L']:5.1f} L   "
              f"({m['weight_kg']:.1f} kg true)")

    print("\nOverall accuracy (mean over all 60 test images):")
    print(f"{'method':<24}{'fill IoU':>10}{'volume err':>12}{'weight err':>12}")
    methods = [("gt", "GT mask (ceiling)"), ("classical", "Classical CV (no AI)")]
    yolo = None
    if args.yolo:
        from ultralytics import YOLO
        yolo = YOLO(os.path.join(HERE, "weights", "synthetic_fill_best.pt"))
        methods.append(("yolo", "YOLO AI (trained)"))
    for key, name in methods:
        io, ev, ew = run(key, manifest, calib, yolo)
        print(f"{name:<24}{io:>10.3f}{ev:>11.1f}%{ew:>11.1f}%")
    print("\n(GT shows the math ceiling; classical is the weak no-AI baseline; "
          "the trained AI ~matches GT — see results/POC_results.ipynb for the full story.)")


if __name__ == "__main__":
    main()
