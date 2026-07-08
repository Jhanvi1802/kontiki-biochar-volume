#!/usr/bin/env python
"""
Analyse the camera-angle sweep: run the estimation pipeline on every viewpoint and
report which camera elevation/azimuth gives the lowest volume error. Outputs a
table + a chart (results/figures/angle_sweep.png).
"""
import os, json, cv2, numpy as np, sys
sys.path.insert(0, os.path.dirname(__file__))
import geometry as G, segment as S
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SW = "synthetic/sweep"
THETA = 38.0   # assumed heap angle in the estimator (the deployed system won't know the true one)


def fill_mask_from_id(idmask):
    b, g, r = idmask[:, :, 0], idmask[:, :, 1], idmask[:, :, 2]
    return ((r > 120) & (g < 90) & (b < 90)).astype(np.uint8) * 255


def main():
    man = json.load(open(f"{SW}/manifest.json"))
    rows = []
    for m in man:
        idm = cv2.imread(f"{SW}/mask/{m['id']:05d}.png")
        rgb = cv2.imread(f"{SW}/rgb/{m['id']:05d}.png")
        gtm = fill_mask_from_id(idm)
        vis_px = int((gtm > 0).sum())
        cam = m["camera"]
        def vol_err(mask):
            h = G.estimate_fill_height_raw(mask, cam["K"], cam["R"], cam["t"],
                                           m["radius_m"], m["height_m"])
            if not np.isfinite(h):
                return np.nan
            V = G.volume_liters(m["radius_m"], h, THETA)
            return abs(V - m["V_mesh_L"]) / m["V_mesh_L"] * 100
        rows.append(dict(el=m["elevation_deg"], az=m["azimuth_deg"], fill=m["fill_height_m"],
                         vis_px=vis_px, err_gt=vol_err(gtm),
                         err_classical=vol_err(S.classical_fill_mask(rgb))))
    # aggregate by elevation (avg over fills + azimuths)
    els = sorted({r["el"] for r in rows})
    agg = []
    for el in els:
        sub = [r for r in rows if r["el"] == el]
        agg.append(dict(elevation=el,
            mean_vis_px=float(np.mean([r["vis_px"] for r in sub])),
            err_gt=float(np.nanmean([r["err_gt"] for r in sub])),
            err_classical=float(np.nanmean([r["err_classical"] for r in sub])),
            n_failed=int(sum(1 for r in sub if not np.isfinite(r["err_gt"])))))
    json.dump(dict(rows=rows, by_elevation=agg),
              open("results/angle_sweep_results.json", "w"), indent=1)

    best = min(agg, key=lambda a: (a["err_gt"] if np.isfinite(a["err_gt"]) else 1e9))
    print("elevation | visible px | vol err (perfect mask) | vol err (classical) | failed")
    for a in agg:
        print(f"  {a['elevation']:>3}°     {a['mean_vis_px']:>8.0f}   "
              f"{a['err_gt']:>6.1f}%               {a['err_classical']:>6.1f}%          {a['n_failed']}")
    print(f"\nBEST elevation ~ {best['elevation']}° "
          f"(lowest volume error {best['err_gt']:.1f}% with a clean mask)")

    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    e = [a["elevation"] for a in agg]
    err = [a["err_gt"] for a in agg]
    ax2 = ax1.twinx()
    ax2.bar(e, [a["mean_vis_px"] for a in agg], width=5, alpha=0.15, color="gray",
            label="charcoal visible")
    ax2.set_ylabel("charcoal pixels the camera can see (grey bars)")
    # recommended band
    ax1.axvspan(45, 70, color="green", alpha=0.10)
    ax1.text(57, 17, "recommended\nmounting band", ha="center", color="green", fontsize=10)
    ax1.plot(e, err, "o-", color="#3366cc", lw=2, label="volume error")
    # flag elevations where some views failed (surface fully occluded)
    for a in agg:
        if a["n_failed"]:
            ax1.annotate(f"{a['n_failed']}/6 views\nblind", (a["elevation"], a["err_gt"]),
                         textcoords="offset points", xytext=(0, 12), ha="center",
                         color="#cc3333", fontsize=8)
    ax1.axhline(5, ls=":", c="green"); ax1.text(78, 5.4, "±5% target", color="green", fontsize=9)
    ax1.set_xlabel("camera elevation (degrees above horizontal)")
    ax1.set_ylabel("volume error %  (lower = better)")
    ax1.set_ylim(0, 20); ax1.set_zorder(2); ax1.patch.set_visible(False)
    ax1.set_title("Where to mount the camera: volume error vs elevation\n"
                  f"(drum r={man[0]['radius_m']*100:.0f}cm H={man[0]['height_m']*100:.0f}cm — "
                  "below ~40° the rim hides the surface)")
    ax1.legend(loc="upper right")
    fig.tight_layout(); fig.savefig("results/figures/angle_sweep.png", dpi=130)
    print("wrote results/figures/angle_sweep.png")


if __name__ == "__main__":
    main()
