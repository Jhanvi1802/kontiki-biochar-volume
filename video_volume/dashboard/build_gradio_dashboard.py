"""Builds KonTiki_Dashboard.ipynb — a ONE-CELL Colab launcher for a real web
dashboard (Gradio): upload a kiln video -> get the biochar volume. No .ply juggling.
Runs on the free Colab GPU and prints a public dashboard link (share=True).
The volume maths is the validated estimate_volume.py, embedded (single source of truth).
Run:  python build_gradio_dashboard.py
"""
import json, os, base64

_EST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "estimate_volume.py")
_ARUCO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aruco_tools.py")
EST_B64 = base64.b64encode(open(_EST, "rb").read()).decode("ascii")
ARUCO_B64 = base64.b64encode(open(_ARUCO, "rb").read()).decode("ascii")
CELLS = []
def md(s):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s})
def code(s): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s})

md("""# Kon-Tiki Biochar Volume — Web Dashboard (upload video → volume)

**One step:** `Runtime → Change runtime type → GPU (T4)`, then **Run this cell**.
After ~1–2 min it prints a **public link** (`…gradio.live`). Open it → **upload a kiln
video → get the volume**. No download / re-upload. Share the link with anyone.

_The GPU reconstruction runs here on Colab's free GPU (this is the only part that needs a
GPU); the volume maths is the same code validated to ~2–4% on ground truth._
""")

CELL = r'''import os, sys, glob, shutil, base64, subprocess, torch, numpy as np, cv2
subprocess.run("pip -q install gradio opencv-python-headless scipy", shell=True)
if not os.path.exists("vggt"):
    subprocess.run("git clone -q https://github.com/facebookresearch/vggt.git", shell=True)
subprocess.run("grep -viE '^(torch|torchvision|torchaudio|numpy)' vggt/requirements.txt > /tmp/r.txt", shell=True)
subprocess.run("pip -q install -r /tmp/r.txt", shell=True)
if "vggt" not in sys.path: sys.path.append("vggt")
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > GPU (T4), then re-run."

open("estimate_volume.py", "w", encoding="utf-8").write(base64.b64decode("__EST_B64__").decode("utf-8"))
open("aruco_tools.py", "w", encoding="utf-8").write(base64.b64decode("__ARUCO_B64__").decode("utf-8"))
from estimate_volume import estimate_points
import aruco_tools
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import gradio as gr

print("loading VGGT model (once)...")
MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()

def extract_frames(video, n=40, out="frames"):
    if os.path.exists(out): shutil.rmtree(out)
    os.makedirs(out)
    cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = 0
        while cap.grab(): total += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    slot = total / n; k = 0
    for i in range(n):
        lo, hi = int(i * slot), int((i + 1) * slot); best = None
        for idx in np.linspace(lo, max(lo, hi - 1), 5).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, fr = cap.read()
            if not ok: continue
            sc = cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if best is None or sc > best[1]: best = (idx, sc, fr)
        if best:
            cv2.imwrite(f"{out}/f_{k:03d}.jpg", best[2], [cv2.IMWRITE_JPEG_QUALITY, 95]); k += 1
    cap.release(); return sorted(glob.glob(f"{out}/*.jpg"))

def save_ply(path, P):
    P = np.asarray(P, np.float32)
    hdr = ("ply\nformat binary_little_endian 1.0\n"
           f"element vertex {len(P)}\n"
           "property float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(P.tobytes())

def reconstruct(paths, marker_cm=0.0):
    dt = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    imgs = load_and_preprocess_images(paths).to("cuda")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dt):
        pred = MODEL(imgs)
    def gk(d, *ks):
        for kk in ks:
            if kk in d: return d[kk]
        raise KeyError(ks)
    extr, intr = pose_encoding_to_extri_intri(gk(pred, "pose_enc"), imgs.shape[-2:])
    depth = gk(pred, "depth", "depth_map"); conf = gk(pred, "depth_conf", "point_conf", "depth_confidence")
    wp = np.asarray(unproject_depth_map_to_point_map(depth.squeeze(0), extr.squeeze(0), intr.squeeze(0)))  # [N,H,W,3]
    cf = conf.squeeze(0).float().cpu().numpy()                    # [N,H,W]

    aruco_scale = None                                            # metric scale from markers (optional)
    if marker_cm and marker_cm > 0:
        try:
            im = imgs.detach().float().cpu().numpy().transpose(0, 2, 3, 1)       # [N,H,W,3]
            u8 = [((a - a.min()) / (a.max() - a.min() + 1e-9) * 255).astype("uint8") for a in im]
            rr = aruco_tools.scale_from_marker(u8, wp, float(marker_cm))
            if rr:
                aruco_scale = rr["scale_cm_per_unit"]; print("ArUco metric scale:", rr)
        except Exception as e:
            print("ArUco scale skipped:", e)

    w = wp.reshape(-1, 3); cff = cf.reshape(-1)
    w = w[(cff >= np.quantile(cff, 0.5)) & np.isfinite(w).all(1)]
    if len(w) > 300000:
        w = w[np.random.default_rng(0).choice(len(w), 300000, replace=False)]
    return w, aruco_scale

def process(video, marker_cm, rim_cm, density):
    if not video:
        return "<p>Please upload a kiln video.</p>", None, None, None
    try:
        paths = extract_frames(video)
        world, aruco_scale = reconstruct(paths, marker_cm)
        save_ply("dense.ply", world)
        res = estimate_points(world, rim_radius_cm=float(rim_cm), density=float(density),
                              scale_cm_per_unit=aruco_scale,
                              views_png="views.png", heatmap_png="heatmap.png")
    except Exception as e:
        return f"<p style='color:#b00'>Could not process this video: {e}</p>", None, None, None
    V, Vf = res["volume_L"], res["volume_L_flatfill"]
    fill = max(0, min(100, res["fill_pct"]))
    conf = res.get("confidence", "?")
    ccol = {"good": "#2e7d33", "medium": "#c60", "low": "#c0392b", "unreliable": "#c0392b"}.get(conf, "#666")
    warns = "".join(f"<li>{w}</li>" for w in res.get("warnings", []))
    warn_html = f"<ul style='color:#c60;margin:6px 0 0;padding-left:18px;font-size:13px'>{warns}</ul>" if warns else ""
    html = f"""<div style="font-family:system-ui,Segoe UI,Arial">
      <div style="font-size:13px;letter-spacing:1px;color:#888">ESTIMATED BIOCHAR VOLUME</div>
      <div style="font-size:54px;font-weight:800;color:#2e7d33;line-height:1">{V:,.0f} L</div>
      <div style="color:#555;margin-top:4px">&#8776; {res['weight_kg']:,.0f} kg &middot; {V/1000:.2f} m&sup3;</div>
      <div style="margin-top:12px;font-size:13px;color:#888">FILL &mdash; {fill:.0f}% of a ~1000 L kiln (height {res['fill_height_cm']:.0f} cm)</div>
      <div style="height:16px;background:#eee;border-radius:9px;overflow:hidden;margin-top:4px">
        <div style="height:100%;width:{fill:.0f}%;background:linear-gradient(90deg,#2d7ef7,#4fe08a)"></div></div>
      <div style="margin-top:12px">cross-check {Vf:,.0f} L &middot; scale from {res.get('scale_source','?')}</div>
      <div style="margin-top:10px;font-weight:700;color:{ccol}">Self-check: {conf.upper()}</div>
      {warn_html}
    </div>"""
    return html, "views.png", "heatmap.png", "dense.ply"

with gr.Blocks(title="Kon-Tiki Biochar Volume") as demo:
    gr.Markdown("# Kon-Tiki Biochar Volume - Video Platform")
    gr.Markdown("Upload a slow-orbit video of the biochar-filled kiln to get the volume. "
                "For TRUE scale, print & measure an ArUco marker and enter its edge size; "
                "otherwise it uses the standard Kon-Tiki rim.")
    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="1 - Upload slow-orbit kiln video")
            mk = gr.Number(value=0.0, label="ArUco marker edge (cm)  -  0 = no marker (use rim)")
            rimn = gr.Number(value=75.0, label="Kiln rim RADIUS (cm)  -  75 for Kon-Tiki 1000")
            dn = gr.Number(value=0.25, label="Biochar density (kg/L)")
            go = gr.Button("Estimate biochar volume", variant="primary")
        with gr.Column():
            res_html = gr.HTML(label="Result")
    with gr.Row():
        img3d = gr.Image(label="3-D reconstruction (top = rim disk, side = cone)")
        imgheat = gr.Image(label="Biochar depth heatmap (volume = sum)")
    plyf = gr.File(label="Download 3-D cloud (dense.ply)")
    go.click(process, [vid, mk, rimn, dn], [res_html, img3d, imgheat, plyf])

print("launching platform - a public https://....gradio.live link will appear below")
demo.launch(share=True)
'''.replace("__EST_B64__", EST_B64).replace("__ARUCO_B64__", ARUCO_B64)

code(CELL)

md("""### ArUco marker = true scale (recommended)
The result shows **"scale from marker"** or **"scale from rim (assumed)"**. To get the size
read from the *video itself* (not assumed), use ArUco markers:
1. Print the markers (`video_volume/markers/aruco_*.png`) on **matte** paper.
2. **Measure the black square's real edge with a ruler** and set `MARKER_CM` in Cell 1 to that value (cm).
3. Lay **3–4 markers flat on the ground** around the kiln, ~90° apart, different IDs (so one is always visible), **after quenching**.
4. Record the slow orbit as usual — the dashboard detects them and scales metrically.
No markers? It falls back to the assumed rim (fine for a standard Kon-Tiki 1000).

### Notes
- The **public link** works from any device for ~72 h while this cell runs.
- Keep this Colab tab open; closing it stops the dashboard. Re-run the cell to restart.
- For an **always-on** dashboard (no Colab), deploy the same app to a **GPU host**
  (Hugging Face Spaces GPU / Modal) — ask and I'll provide `gradio_app.py` + steps.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KonTiki_Dashboard.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "|", len(CELLS), "cells")
