"""Builds KonTiki_Video_to_Volume.ipynb — streamlined Colab GPU app:
  * Cell 1 (SETUP, run once): install + load VGGT + embed the tested volume engine.
  * Cell 2 (RUN, once per kiln): upload a video -> frames -> dense 3-D -> litres.
The volume maths is the SAME validated estimate_volume.py (embedded, not duplicated).
Run:  python build_notebook.py
"""
import json, os, base64

_EST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "estimate_volume.py")
EST_B64 = base64.b64encode(open(_EST, "rb").read()).decode("ascii")
CELLS = []
def md(src):   CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src})
def code(src): CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src})

md("""# Kon-Tiki Biochar Volume — Video → Litres  (Colab GPU app)

**How to use:**
1. `Runtime → Change runtime type → GPU (T4)`
2. Run **Cell 1 (Setup)** once — installs everything and loads the model.
3. Run **Cell 2 (Measure a kiln)** — upload a slow-orbit video → get the volume.
   Re-run Cell 2 for each new kiln (Setup stays loaded).

> Reconstruction quality depends on capture — follow the video SOP (slow full circle,
> tilt ~50–60° down into the kiln, full rim always visible, 1080p+). The volume maths is
> validated to ~2–4% on ground truth; a proper video + a known-volume kiln proves the rest.
""")

md("### Cell 1 · Setup — run once")
code(f"""import os, sys, shutil, glob, base64, torch, numpy as np, cv2
# deps (keep Colab's matched torch/torchvision/numpy -> avoids nms/numpy breakage)
!pip -q install opencv-python-headless scipy 2>/dev/null
if not os.path.exists('vggt'):
    !git clone -q https://github.com/facebookresearch/vggt.git
!grep -viE '^(torch|torchvision|torchaudio|numpy)' vggt/requirements.txt > /tmp/r.txt
!pip -q install -r /tmp/r.txt 2>/dev/null
if 'vggt' not in sys.path: sys.path.append('vggt')
assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > GPU (T4), then re-run."

# the tested volume engine (embedded, byte-identical to estimate_volume.py)
open("estimate_volume.py", "w", encoding="utf-8").write(base64.b64decode("{EST_B64}").decode("utf-8"))
from estimate_volume import estimate_points
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

MODEL = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
import torchvision
print("Setup ready | GPU:", torch.cuda.get_device_name(0),
      "| torch", torch.__version__, "| torchvision", torchvision.__version__)

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
        lo, hi = int(i*slot), int((i+1)*slot); best = None
        for idx in np.linspace(lo, max(lo, hi-1), 5).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); ok, fr = cap.read()
            if not ok: continue
            s = cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if best is None or s > best[1]: best = (idx, s, fr)
        if best:
            cv2.imwrite(f"{{out}}/f_{{k:03d}}.jpg", best[2], [cv2.IMWRITE_JPEG_QUALITY, 95]); k += 1
    cap.release(); return sorted(glob.glob(f"{{out}}/*.jpg"))

def reconstruct(paths):
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
    w = np.asarray(unproject_depth_map_to_point_map(depth.squeeze(0), extr.squeeze(0), intr.squeeze(0))).reshape(-1, 3)
    conf = conf.squeeze(0).float().cpu().numpy().reshape(-1)
    w = w[(conf >= np.quantile(conf, 0.5)) & np.isfinite(w).all(1)]
    if len(w) > 300000:
        w = w[np.random.default_rng(0).choice(len(w), 300000, replace=False)]
    return w

def save_ply(path, P):
    P = np.asarray(P, np.float32)
    hdr = ("ply\\nformat binary_little_endian 1.0\\n"
           f"element vertex {{len(P)}}\\n"
           "property float x\\nproperty float y\\nproperty float z\\nend_header\\n")
    with open(path, "wb") as f:
        f.write(hdr.encode()); f.write(P.tobytes())
print("helpers ready: extract_frames(), reconstruct(), estimate_points()")""")

md("""### Cell 2 · Measure a kiln — upload a video, get the volume
Run this cell, pick your slow-orbit video, and wait. It extracts frames, builds the
3-D model, and prints the biochar volume. Re-run it for each new kiln.""")
code("""from google.colab import files
from IPython.display import Image, display

up = files.upload()                       # pick your slow-orbit .mp4/.mov
VIDEO = list(up.keys())[0]
print("video:", VIDEO)

print("1/3 extracting frames..."); paths = extract_frames(VIDEO); print("   ", len(paths), "frames")
print("2/3 reconstructing 3-D (GPU)..."); world = reconstruct(paths); print("   ", len(world), "points")
print("3/3 measuring volume...")
save_ply("dense.ply", world)
res = estimate_points(world, rim_radius_cm=75.0, views_png="kiln_views.png", heatmap_png="kiln_heatmap.png")

print("\\n" + "=" * 46)
print(f"  BIOCHAR VOLUME : {res['volume_L']:6.0f} L    (~{res['weight_kg']:.0f} kg)")
print(f"  fill level     : {res['fill_height_cm']:.0f} cm  ({res['fill_pct']:.0f}% of a ~1000 L kiln)")
print(f"  cross-check    : {res['volume_L_flatfill']:6.0f} L   (should be close to the volume)")
print(f"  SELF-CHECK     : {res['confidence'].upper()}")
for w in res.get("warnings", []):
    print(f"    ! {w}")
print("=" * 46)
display(Image("kiln_views.png"))          # TOP = rim disk, SIDE = cone/bowl
display(Image("kiln_heatmap.png"))        # biochar depth heatmap (volume = sum)""")

md("""### Notes
- **Rim scale:** assumes a standard Kon-Tiki 1000 rim (Ø150 cm). For other kilns, change
  `rim_radius_cm`, or lay a 1-metre marker in the video.
- **Trust check:** if the two volume numbers disagree by more than ~8%, or the 3-D views
  don't look like a clean bowl, the capture was poor — re-shoot per the SOP.
- **Downloads:** `dense.ply` (3-D model) and `kiln_views.png` are saved in the file browser.
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KonTiki_Video_to_Volume.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "|", len(CELLS), "cells")
