#!/usr/bin/env python
"""
Track A evaluation on the real CORSMAL C-CCM benchmark (granular fillings only:
rice + pasta = charcoal proxies), within the provided container bbox ROI.

C-CCM has NO capacity/mass ground truth, so absolute volume/weight are not
scoreable here (they come from Track B). What IS scoreable on real images:
  CONTAINER SEGMENTATION IoU  vs the provided (binarised) Mask R-CNN masks,
  comparing  classical CV (no training)  vs  a YOLO11-seg model trained on
  real crops, evaluated on UNSEEN containers (container-disjoint split).

This demonstrates the Stage-1 segmenter survives real, cluttered images.
"""
import os, json, argparse, shutil, cv2, numpy as np, sys
sys.path.insert(0, os.path.dirname(__file__))
import segment as S
from ccm_dataset import CCM, make_splits

MARGIN = 0.12


def crop_box(rec, W, H):
    x1, y1, x2, y2 = rec["bbox"]
    bw, bh = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - bw * MARGIN)); cy1 = max(0, int(y1 - bh * MARGIN))
    cx2 = min(W, int(x2 + bw * MARGIN)); cy2 = min(H, int(y2 + bh * MARGIN))
    return cx1, cy1, cx2, cy2


def build_crops(ds, ids, out, sp):
    os.makedirs(f"{out}/images/{sp}", exist_ok=True)
    os.makedirs(f"{out}/labels/{sp}", exist_ok=True)
    kept = []
    for i in ids:
        rgb = cv2.imread(ds.rgb_path(i)); cm = ds.container_mask(i)
        if rgb is None or cm is None:
            continue
        H, W = rgb.shape[:2]
        x1, y1, x2, y2 = crop_box(ds.record(i), W, H)
        rc = rgb[y1:y2, x1:x2]; gc = cm[y1:y2, x1:x2]
        if rc.size == 0 or (gc > 0).sum() < 50:
            continue
        cv2.imwrite(f"{out}/images/{sp}/{i:06d}.png", rc)
        cnts, _ = cv2.findContours((gc > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        ch, cw = gc.shape
        with open(f"{out}/labels/{sp}/{i:06d}.txt", "w") as f:
            for c in cnts:
                if cv2.contourArea(c) < 30:
                    continue
                ap = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True)
                ap = ap.reshape(-1, 2).astype(float)
                if len(ap) < 3:
                    continue
                ap[:, 0] /= cw; ap[:, 1] /= ch
                f.write("0 " + " ".join(f"{v:.6f}" for v in ap.flatten()) + "\n")
        kept.append(i)
    return kept


def classical_iou(ds, ids):
    ious = []
    for i in ids:
        rgb = cv2.imread(ds.rgb_path(i)); cm = ds.container_mask(i)
        if rgb is None or cm is None:
            continue
        H, W = rgb.shape[:2]
        x1, y1, x2, y2 = crop_box(ds.record(i), W, H)
        pred = S.classical_container_mask(rgb[y1:y2, x1:x2])
        ious.append(S.iou(pred, cm[y1:y2, x1:x2]))
    return ious


def yolo_iou(weights, ds, ids):
    from ultralytics import YOLO
    model = YOLO(weights)
    ious = []
    for i in ids:
        rgb = cv2.imread(ds.rgb_path(i)); cm = ds.container_mask(i)
        if rgb is None or cm is None:
            continue
        H, W = rgb.shape[:2]
        x1, y1, x2, y2 = crop_box(ds.record(i), W, H)
        rc = rgb[y1:y2, x1:x2]; gc = cm[y1:y2, x1:x2]
        res = model.predict(rc, verbose=False, device=0, conf=0.001, imgsz=320,
                            retina_masks=True)[0]
        out = np.zeros(gc.shape, np.uint8)
        if res.masks is not None and len(res.masks.data):
            # keep the single highest-confidence detection (the container)
            best = int(np.argmax(res.boxes.conf.cpu().numpy()))
            md = res.masks.data.cpu().numpy()[best]
            out = (cv2.resize(md, (gc.shape[1], gc.shape[0])) > 0.5).astype(np.uint8)
        ious.append(S.iou(out, gc))
    return ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/ccm")
    ap.add_argument("--out", default="results/ccm_results.json")
    ap.add_argument("--yolo_ds", default="data/ccm/yolo_crops")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--train_yolo", action="store_true")
    args = ap.parse_args()

    ds = CCM(args.root)
    splits = make_splits(ds)
    gran = {sp: [i for i in splits[sp] if ds.ann[i]["filling_type"] in (1, 2)]
            for sp in ("train", "val", "test")}
    print("granular split sizes:", {k: len(v) for k, v in gran.items()},
          "| test containers", splits["_meta"]["test_containers"])

    res = dict(track="A_real_CCM", splits=splits["_meta"],
               granular_split_sizes={k: len(v) for k, v in gran.items()},
               note="No capacity/mass GT in C-CCM; absolute volume/weight come "
                    "from Track B. Here we score container segmentation IoU on "
                    "unseen containers (classical vs learned).")

    # classical baseline
    cls_iou = classical_iou(ds, gran["test"])
    res["classical_container"] = dict(n=len(cls_iou),
        mean_iou=float(np.mean(cls_iou)), median_iou=float(np.median(cls_iou)),
        frac_above_0p5=float(np.mean(np.array(cls_iou) > 0.5)))
    print("classical container IoU: mean=%.3f median=%.3f" %
          (res["classical_container"]["mean_iou"], res["classical_container"]["median_iou"]))

    # learned YOLO
    if args.train_yolo:
        for sp in ("train", "val", "test"):
            build_crops(ds, gran[sp], args.yolo_ds, sp)
        yaml = (f"path: {os.path.abspath(args.yolo_ds)}\ntrain: images/train\n"
                f"val: images/val\ntest: images/test\nnc: 1\nnames: [container]\n")
        open(f"{args.yolo_ds}/data.yaml", "w").write(yaml)
        from ultralytics import YOLO
        model = YOLO("yolo11n-seg.pt")
        model.train(data=f"{args.yolo_ds}/data.yaml", epochs=args.epochs, imgsz=320,
                    batch=32, device=0, name="contseg", workers=2, seed=0,
                    exist_ok=True, verbose=False, plots=False)
        wp = str(model.trainer.best)
        open(f"{args.yolo_ds}/best.txt", "w").write(wp)
        y_iou = yolo_iou(wp, ds, gran["test"])
        res["yolo_container"] = dict(n=len(y_iou), weights=wp,
            mean_iou=float(np.mean(y_iou)), median_iou=float(np.median(y_iou)),
            frac_above_0p5=float(np.mean(np.array(y_iou) > 0.5)),
            frac_above_0p9=float(np.mean(np.array(y_iou) > 0.9)))
        print("YOLO container IoU: mean=%.3f median=%.3f frac>0.9=%.2f" %
              (res["yolo_container"]["mean_iou"], res["yolo_container"]["median_iou"],
               res["yolo_container"]["frac_above_0p9"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
