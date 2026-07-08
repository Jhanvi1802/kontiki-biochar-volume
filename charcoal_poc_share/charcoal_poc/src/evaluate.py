#!/usr/bin/env python
"""
Track B end-to-end evaluation: segmentation IoU + volume/weight accuracy on the
held-out TEST split, for each Stage-1 method (gt / classical / yolo).

Pipeline per method:
  mask -> robust fill height (calibrated on TRAIN) -> volume (cyl, cyl+heap)
       -> weight (known density). Scored vs exact synthetic ground truth.
"""
import os, json, argparse, cv2, numpy as np, sys
sys.path.insert(0, os.path.dirname(__file__))
import segment as S, geometry as G

THETA_ASSUMED = 38.0   # nominal angle of repose used for heap correction


def mape(est, true):
    est, true = np.asarray(est, float), np.asarray(true, float)
    m = np.isfinite(est) & np.isfinite(true) & (true > 0)
    return float(np.mean(np.abs(est[m] - true[m]) / true[m]) * 100)


def load_masks(renders, i, method, yolo_model):
    fn = f"{i:05d}.png"
    if method == "gt":
        return cv2.imread(f"{renders}/fill_mask/{fn}", 0)
    rgb = cv2.imread(f"{renders}/rgb/{fn}")
    if method == "classical":
        return S.classical_fill_mask(rgb)
    if method == "yolo":
        res = yolo_model.predict(rgb, verbose=False, device=0, retina_masks=True)[0]
        H, W = rgb.shape[:2]
        out = np.zeros((H, W), np.uint8)
        if res.masks is not None:
            for m in res.masks.data.cpu().numpy():
                out |= (cv2.resize(m, (W, H)) > 0.5).astype(np.uint8) * 255
        return out
    raise ValueError(method)


def raw_height(renders, i, mask, manifest_by_id):
    m = manifest_by_id[i]; cam = m["camera"]
    return G.estimate_fill_height_raw(mask, cam["K"], cam["R"], cam["t"],
                                      m["radius_m"], m["height_m"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default="synthetic/renders")
    ap.add_argument("--methods", nargs="+", default=["gt", "classical", "yolo"])
    ap.add_argument("--out", default="results/synthetic_results.json")
    args = ap.parse_args()

    manifest = json.load(open(f"{args.renders}/manifest.json"))
    by_id = {m["id"]: m for m in manifest}
    splits = json.load(open(f"{args.renders}/yolo_splits.json"))
    train, test = splits["train"], splits["test"]

    yolo_model = None
    if "yolo" in args.methods:
        bp = f"{args.renders}/yolo_best.txt"
        wp = open(bp).read().strip() if os.path.exists(bp) else ""
        if wp and os.path.exists(wp):
            from ultralytics import YOLO
            yolo_model = YOLO(wp)
        else:
            print("!! YOLO weights missing, skipping yolo method")
            args.methods = [m for m in args.methods if m != "yolo"]

    results = {}
    for method in args.methods:
        # --- calibration on TRAIN
        hr_tr, ht_tr = [], []
        for i in train:
            mk = load_masks(args.renders, i, method, yolo_model)
            hr_tr.append(raw_height(args.renders, i, mk, by_id))
            ht_tr.append(by_id[i]["fill_height_m"])
        ab = G.fit_linear(hr_tr, ht_tr)

        # --- evaluate on TEST
        rows = []
        for i in test:
            m = by_id[i]
            mk = load_masks(args.renders, i, method, yolo_model)
            gt = cv2.imread(f"{args.renders}/fill_mask/{i:05d}.png", 0)
            io = S.iou(mk, gt)
            hr = raw_height(args.renders, i, mk, by_id)
            hcal = float(G.apply_linear(hr, ab)) if np.isfinite(hr) else np.nan
            r = m["radius_m"]
            v_cyl = G.volume_liters(r, hcal) if np.isfinite(hcal) else np.nan
            v_heap = G.volume_liters(r, hcal, THETA_ASSUMED) if np.isfinite(hcal) else np.nan
            w_est = G.weight_kg(v_heap, m["bulk_density_kgm3"]) if np.isfinite(v_heap) else np.nan
            rows.append(dict(id=i, iou=io, h_true=m["fill_height_m"], h_est=hcal,
                             V_true=m["V_mesh_L"], V_cyl=v_cyl, V_heap=v_heap,
                             W_true=m["weight_kg"], W_est=w_est,
                             fill_fraction=m["fill_fraction"], form=m["material_form"]))
        rr = rows
        valid = [r for r in rr if np.isfinite(r["h_est"])]
        summ = dict(
            calib_ab=ab, n_test=len(rr), n_valid=len(valid),
            mean_iou=float(np.mean([r["iou"] for r in rr])),
            height_mae_mm=float(np.mean([abs(r["h_est"]-r["h_true"]) for r in valid])*1000),
            vol_mape_cyl=mape([r["V_cyl"] for r in rr], [r["V_true"] for r in rr]),
            vol_mape_heap=mape([r["V_heap"] for r in rr], [r["V_true"] for r in rr]),
            weight_mape=mape([r["W_est"] for r in rr], [r["W_true"] for r in rr]),
        )
        # breakdown by fill level and form (using heap-corrected volume)
        def grp(key, fn):
            out = {}
            for g in sorted({fn(r) for r in valid}):
                sub = [r for r in valid if fn(r) == g]
                out[str(g)] = dict(n=len(sub),
                    vol_mape_heap=mape([r["V_heap"] for r in sub], [r["V_true"] for r in sub]),
                    mean_iou=float(np.mean([r["iou"] for r in sub])))
            return out
        summ["by_form"] = grp("form", lambda r: r["form"])
        summ["by_fill_band"] = grp("band", lambda r: ("low" if r["fill_fraction"] < 0.55
                                                      else "high"))
        results[method] = dict(summary=summ, rows=rr)
        print(f"[{method}] IoU={summ['mean_iou']:.3f} "
              f"h_MAE={summ['height_mae_mm']:.1f}mm "
              f"vol_MAPE(cyl)={summ['vol_mape_cyl']:.1f}% "
              f"vol_MAPE(+heap)={summ['vol_mape_heap']:.1f}% "
              f"weight_MAPE={summ['weight_mape']:.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
