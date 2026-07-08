#!/usr/bin/env python
"""
Postprocess the Blender ID-mask renders into the binary masks the estimation
pipeline consumes, plus QA overlays and a manifest consistency check.

Blender writes:
  renders/rgb/{id}.png    beauty
  renders/mask/{id}.png   flat class colours  charcoal=red drum=green marker=blue

This script writes:
  renders/fill_mask/{id}.png   binary charcoal (fill) mask  (255/0)
  renders/drum_mask/{id}.png   binary container mask        (255/0)
  renders/qa/{id}.png          rgb with mask contours + GT text   (sampled)

and prints/saves a summary (analytic-vs-mesh volume agreement, mask areas).
"""
import os, json, argparse, cv2, numpy as np

def cls_masks(mask_bgr):
    b, g, r = mask_bgr[:, :, 0], mask_bgr[:, :, 1], mask_bgr[:, :, 2]
    charcoal = ((r > 120) & (g < 90) & (b < 90)).astype(np.uint8) * 255
    drum = ((g > 120) & (r < 90) & (b < 90)).astype(np.uint8) * 255
    marker = ((b > 120) & (r < 90) & (g < 90)).astype(np.uint8) * 255
    return charcoal, drum, marker

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", default="synthetic/renders")
    ap.add_argument("--qa_every", type=int, default=1, help="write a QA overlay every N samples")
    args = ap.parse_args()
    R = args.renders
    for sub in ("fill_mask", "drum_mask", "qa"):
        os.makedirs(os.path.join(R, sub), exist_ok=True)

    manifest = json.load(open(os.path.join(R, "manifest.json")))
    by_id = {m["id"]: m for m in manifest}
    rows = []
    for m in manifest:
        i = m["id"]
        fn = f"{i:05d}.png"
        mask = cv2.imread(os.path.join(R, "mask", fn))
        rgb = cv2.imread(os.path.join(R, "rgb", fn))
        if mask is None or rgb is None:
            print(f"  !! missing render for {i}"); continue
        char, drum, marker = cls_masks(mask)
        cv2.imwrite(os.path.join(R, "fill_mask", fn), char)
        cv2.imwrite(os.path.join(R, "drum_mask", fn), drum)

        char_px = int((char > 0).sum())
        drum_px = int((drum > 0).sum())
        v_err = abs(m["V_mesh_L"] - m["V_analytic_L"]) / m["V_analytic_L"] * 100
        rows.append(dict(id=i, charcoal_px=char_px, drum_px=drum_px,
                         marker_px=int((marker > 0).sum()),
                         V_mesh_L=m["V_mesh_L"], V_analytic_L=m["V_analytic_L"],
                         mesh_vs_analytic_pct=v_err,
                         weight_kg=m["weight_kg"], fill_fraction=m["fill_fraction"]))

        if i % args.qa_every == 0:
            ov = rgb.copy()
            for msk, col in ((char, (0, 0, 255)), (drum, (0, 255, 0)), (marker, (255, 0, 0))):
                cnts, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov, cnts, -1, col, 2)
            txt = [f"id {i}  {m['material_form']}",
                   f"V={m['V_mesh_L']:.1f} L  W={m['weight_kg']:.1f} kg",
                   f"r={m['radius_m']*100:.0f}cm h={m['fill_height_m']*100:.0f}cm "
                   f"fill={m['fill_fraction']*100:.0f}%"]
            for k, t in enumerate(txt):
                cv2.putText(ov, t, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(ov, t, (8, 20 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(R, "qa", fn), ov)

    # summary
    arr = np.array([row["mesh_vs_analytic_pct"] for row in rows])
    char_arr = np.array([row["charcoal_px"] for row in rows])
    summary = dict(
        n=len(rows),
        mesh_vs_analytic_pct_mean=float(arr.mean()),
        mesh_vs_analytic_pct_max=float(arr.max()),
        charcoal_px_min=int(char_arr.min()),
        charcoal_px_mean=float(char_arr.mean()),
        n_zero_charcoal=int((char_arr == 0).sum()),
        V_mesh_L_range=[float(min(r["V_mesh_L"] for r in rows)),
                        float(max(r["V_mesh_L"] for r in rows))],
        weight_kg_range=[float(min(r["weight_kg"] for r in rows)),
                         float(max(r["weight_kg"] for r in rows))],
    )
    json.dump(dict(summary=summary, rows=rows),
              open(os.path.join(R, "qa_report.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
    if summary["n_zero_charcoal"]:
        print(f"WARNING: {summary['n_zero_charcoal']} samples have NO visible charcoal")
    if summary["mesh_vs_analytic_pct_max"] > 2.0:
        print(f"WARNING: mesh vs analytic volume disagree by up to "
              f"{summary['mesh_vs_analytic_pct_max']:.2f}% (displacement too high?)")

if __name__ == "__main__":
    main()
