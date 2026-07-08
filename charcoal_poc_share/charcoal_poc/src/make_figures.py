#!/usr/bin/env python
"""Generate result figures for the POC report from the saved results JSON."""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "results"
os.makedirs(f"{RES}/figures", exist_ok=True)


def fig_pred_vs_true():
    r = json.load(open(f"{RES}/synthetic_results.json"))
    methods = [m for m in ("gt", "yolo", "classical") if m in r]
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.6), squeeze=False)
    for ax, m in zip(axes[0], methods):
        rows = r[m]["rows"]
        vt = np.array([x["V_true"] for x in rows])
        ve = np.array([x["V_heap"] for x in rows], float)
        ok = np.isfinite(ve)
        ax.scatter(vt[ok], ve[ok], s=18, alpha=0.6, edgecolor="k", linewidth=0.3)
        lim = [0, max(vt.max(), np.nanmax(ve)) * 1.05]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("true volume (L)"); ax.set_ylabel("estimated volume (L)")
        s = r[m]["summary"]
        ax.set_title(f"{m}: IoU={s['mean_iou']:.2f}, "
                     f"vol MAPE={s['vol_mape_heap']:.1f}%")
        ax.grid(alpha=0.3)
    fig.suptitle("Track B (synthetic) — estimated vs true volume (cyl+heap)")
    fig.tight_layout()
    fig.savefig(f"{RES}/figures/vol_pred_vs_true.png", dpi=130)
    print("wrote vol_pred_vs_true.png")


def fig_method_bars():
    r = json.load(open(f"{RES}/synthetic_results.json"))
    methods = [m for m in ("gt", "yolo", "classical") if m in r]
    iou = [r[m]["summary"]["mean_iou"] for m in methods]
    vmape = [r[m]["summary"]["vol_mape_heap"] for m in methods]
    wmape = [r[m]["summary"]["weight_mape"] for m in methods]
    x = np.arange(len(methods)); w = 0.6
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].bar(x, iou, w, color="#4477aa")
    ax[0].set_xticks(x); ax[0].set_xticklabels(methods); ax[0].set_ylim(0, 1)
    ax[0].set_title("Mask IoU (fill region)"); ax[0].axhline(0.9, ls="--", c="g", lw=1)
    ax[0].grid(alpha=0.3, axis="y")
    ax[1].bar(x - 0.2, vmape, 0.4, label="volume MAPE", color="#ee6677")
    ax[1].bar(x + 0.2, wmape, 0.4, label="weight MAPE", color="#ccbb44")
    ax[1].set_xticks(x); ax[1].set_xticklabels(methods)
    ax[1].axhline(5, ls="--", c="g", lw=1); ax[1].axhline(10, ls="--", c="orange", lw=1)
    ax[1].set_title("Volume / weight error (%)"); ax[1].legend(); ax[1].grid(alpha=0.3, axis="y")
    fig.suptitle("Track B (synthetic) — accuracy by segmentation method")
    fig.tight_layout()
    fig.savefig(f"{RES}/figures/method_comparison.png", dpi=130)
    print("wrote method_comparison.png")


def fig_track_a():
    if not os.path.exists(f"{RES}/ccm_results.json"):
        return
    r = json.load(open(f"{RES}/ccm_results.json"))
    labels, vals = [], []
    if "classical_container" in r:
        labels.append("classical"); vals.append(r["classical_container"]["mean_iou"])
    if "yolo_container" in r:
        labels.append("YOLO11-seg"); vals.append(r["yolo_container"]["mean_iou"])
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(5, 4.4))
    ax.bar(labels, vals, 0.6, color=["#aa3377", "#228833"])
    ax.set_ylim(0, 1); ax.axhline(0.9, ls="--", c="g", lw=1)
    ax.set_ylabel("container mask IoU")
    ax.set_title("Track A (real C-CCM) — container IoU\n(unseen containers)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(f"{RES}/figures/track_a_iou.png", dpi=130)
    print("wrote track_a_iou.png")


if __name__ == "__main__":
    fig_pred_vs_true()
    fig_method_bars()
    fig_track_a()
