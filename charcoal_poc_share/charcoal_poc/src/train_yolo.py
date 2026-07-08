#!/usr/bin/env python
"""
Method 1: YOLO11-seg for charcoal fill-region segmentation on the synthetic
track. Exports the GT fill masks to YOLO-seg polygon format with a
container-disjoint... (here image-disjoint) train/val/test split, trains
YOLO11n-seg, and reports mask IoU on the held-out TEST split.

Usage:
  python src/train_yolo.py --renders synthetic/renders --epochs 60
"""
import os, json, argparse, shutil, cv2, numpy as np


def mask_to_polys(mask, eps_frac=0.004, min_area=40):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape
    polys = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        eps = eps_frac * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(float)
        if len(ap) < 3:
            continue
        ap[:, 0] /= W; ap[:, 1] /= H
        polys.append(ap)
    return polys


def split_ids(ids, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(ids); rng.shuffle(ids)
    n = len(ids); a = int(0.7 * n); b = int(0.85 * n)
    return dict(train=sorted(ids[:a]), val=sorted(ids[a:b]), test=sorted(ids[b:]))


def export(renders, out, splits):
    for sp in ("train", "val", "test"):
        os.makedirs(f"{out}/images/{sp}", exist_ok=True)
        os.makedirs(f"{out}/labels/{sp}", exist_ok=True)
    for sp, ids in splits.items():
        for i in ids:
            fn = f"{i:05d}.png"
            shutil.copy(f"{renders}/rgb/{fn}", f"{out}/images/{sp}/{fn}")
            mask = cv2.imread(f"{renders}/fill_mask/{fn}", 0)
            polys = mask_to_polys(mask)
            with open(f"{out}/labels/{sp}/{i:05d}.txt", "w") as f:
                for p in polys:
                    f.write("0 " + " ".join(f"{v:.6f}" for v in p.flatten()) + "\n")
    yaml = (f"path: {os.path.abspath(out)}\ntrain: images/train\n"
            f"val: images/val\ntest: images/test\nnc: 1\nnames: [charcoal]\n")
    open(f"{out}/data.yaml", "w").write(yaml)
    return f"{out}/data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default="synthetic/renders")
    ap.add_argument("--out", default="synthetic/yolo_ds")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    args = ap.parse_args()

    manifest = json.load(open(f"{args.renders}/manifest.json"))
    ids = [m["id"] for m in manifest]
    splits = split_ids(ids, seed=0)
    json.dump(splits, open(f"{args.renders}/yolo_splits.json", "w"), indent=1)
    data_yaml = export(args.renders, args.out, splits)
    print("split sizes:", {k: len(v) for k, v in splits.items()})

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=data_yaml, epochs=args.epochs, imgsz=args.imgsz,
                batch=16, device=0, name="fillseg", exist_ok=True,
                workers=2, seed=0, verbose=False, plots=False)
    best = str(model.trainer.best)        # actual best.pt path
    open(f"{args.renders}/yolo_best.txt", "w").write(best)
    metrics = model.val(data=data_yaml, split="test", device=0,
                        name="fillseg_test", exist_ok=True, verbose=False)
    print("TEST mask mAP50:", float(metrics.seg.map50),
          "mask mAP50-95:", float(metrics.seg.map))
    print("best weights:", best, "exists:", os.path.exists(best))


if __name__ == "__main__":
    main()
