#!/usr/bin/env python
"""Render the Results section (markdown) from the saved result JSONs and splice
it into docs/POC_REPORT.md at the <!--RESULTS--> marker."""
import json, os

def f(x, n=1): return f"{x:.{n}f}"

def synthetic_md():
    p = "results/synthetic_results.json"
    if not os.path.exists(p):
        return "_Track B results not available._\n"
    r = json.load(open(p))
    order = [m for m in ("gt", "yolo", "classical") if m in r]
    names = {"gt": "GT mask (geometry ceiling)", "yolo": "YOLO11-seg (learned)",
             "classical": "Classical CV (baseline)"}
    out = ["### 4.1 Track B — synthetic, absolute accuracy (held-out test split)\n",
           "| Segmentation | fill IoU | height MAE | volume MAPE (cyl) | volume MAPE (+heap) | weight MAPE |",
           "|---|---|---|---|---|---|"]
    for m in order:
        s = r[m]["summary"]
        out.append(f"| {names[m]} | {f(s['mean_iou'],3)} | {f(s['height_mae_mm'])} mm "
                   f"| {f(s['vol_mape_cyl'])}% | **{f(s['vol_mape_heap'])}%** | {f(s['weight_mape'])}% |")
    # condition breakdown for the best learned method (or gt)
    key = "yolo" if "yolo" in r else order[0]
    s = r[key]["summary"]
    out.append(f"\n**Breakdown ({names[key]}), heaped volume MAPE:**\n")
    out.append("| condition | n | volume MAPE | IoU |")
    out.append("|---|---|---|---|")
    for grp, d in (("form", s.get("by_form", {})), ("fill", s.get("by_fill_band", {}))):
        for k, v in d.items():
            out.append(f"| {grp}={k} | {v['n']} | {f(v['vol_mape_heap'])}% | {f(v['mean_iou'],3)} |")
    n = r[order[0]]["summary"]["n_test"]
    out.append(f"\n_Test split: {n} held-out images. Calibration fit on the train split only._\n")
    return "\n".join(out) + "\n"

def ccm_md():
    p = "results/ccm_results.json"
    if not os.path.exists(p):
        return "_Track A results not available._\n"
    r = json.load(open(p))
    out = ["### 4.2 Track A — real CORSMAL C-CCM, container segmentation (unseen containers)\n",
           "| Method | container IoU (mean) | median | frac > 0.5 | frac > 0.9 |",
           "|---|---|---|---|---|"]
    c = r.get("classical_container")
    if c:
        out.append(f"| Classical CV (no training) | {f(c['mean_iou'],3)} | {f(c['median_iou'],3)} "
                   f"| {f(c['frac_above_0p5'],2)} | – |")
    y = r.get("yolo_container")
    if y:
        out.append(f"| YOLO11n-seg (learned) | **{f(y['mean_iou'],3)}** | {f(y['median_iou'],3)} "
                   f"| {f(y['frac_above_0p5'],2)} | {f(y['frac_above_0p9'],2)} |")
    ss = r.get("granular_split_sizes", {})
    out.append(f"\n_Granular (rice/pasta) frames — train {ss.get('train','?')}, "
               f"val {ss.get('val','?')}, test {ss.get('test','?')}; test = unseen "
               f"containers {r['splits']['test_containers']}._\n")
    out.append("\n_No capacity/mass GT in C-CCM, so absolute volume/weight are not "
               "scoreable on the real track — they come from Track B (§4.1)._\n")
    return "\n".join(out) + "\n"

def main():
    block = "## 4. Results\n\n" + synthetic_md() + "\n" + ccm_md() + \
            "\n![volume](../results/figures/vol_pred_vs_true.png)\n" + \
            "![methods](../results/figures/method_comparison.png)\n" + \
            "![tracka](../results/figures/track_a_iou.png)\n"
    rp = "docs/POC_REPORT.md"
    txt = open(rp).read()
    a = txt.index("## 4. Results")
    b = txt.index("## 5. Limitations")
    txt = txt[:a] + block + "\n---\n\n" + txt[b:]
    open(rp, "w").write(txt)
    print("report results section updated")
    print(synthetic_md()); print(ccm_md())

if __name__ == "__main__":
    main()
