"""Kon-Tiki Biochar Volume — VIDEO pipeline dashboard (local, CPU).

Runs the model connectivity locally: upload the 3-D point cloud (dense.ply from the
GPU reconstruction step) and this dashboard computes the biochar volume with the
validated estimate_volume engine, and shows the 3-D views, fill gauge, a trust check
and the full pipeline. It also accepts a video and does the local frame-extraction
step, then hands off the single GPU step (reconstruction) which needs Colab / a GPU.

Run:  python app_video.py   ->  http://127.0.0.1:5001
"""
import io, os, sys, base64, glob, shutil, tempfile
import numpy as np, cv2
from flask import Flask, request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))          # import the volume engine
from estimate_volume import estimate_points, R_CM, RB_CM, H_CM, DENSITY

WORK = os.path.join(HERE, "_work"); os.makedirs(WORK, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024   # allow big .ply / video


def load_ply(path):
    """Load a point cloud (.ply, any format) via open3d (installed locally)."""
    import open3d as o3d
    return np.asarray(o3d.io.read_point_cloud(path).points)


def analyse_ply(P):
    tmp = os.path.join(WORK, "views.png")
    res = estimate_points(P, rim_radius_cm=75.0, views_png=tmp)
    with open(tmp, "rb") as f:
        views = base64.b64encode(f.read()).decode()
    Vfull = (1/3) * np.pi * H_CM * (RB_CM**2 + RB_CM*R_CM + R_CM**2) / 1000.0
    return res, views, Vfull


def extract_frames_montage(video_path, n=24):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        total = 0
        while cap.grab(): total += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    slot = max(1, total // n); shots = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * slot))
        ok, fr = cap.read()
        if ok: shots.append(cv2.resize(fr, (160, 90)))
    cap.release()
    if not shots: return 0, None
    cols = 6; rows = (len(shots) + cols - 1) // cols
    canvas = np.zeros((rows * 90, cols * 160, 3), np.uint8)
    for i, s in enumerate(shots):
        r, c = divmod(i, cols); canvas[r*90:r*90+90, c*160:c*160+160] = s
    _, buf = cv2.imencode(".jpg", canvas)
    return total, base64.b64encode(buf).decode()


CSS = """
*{box-sizing:border-box} body{font-family:system-ui,-apple-system,Segoe UI,Arial;margin:0;
 background:radial-gradient(1200px 600px at 50% -10%,#1b2230,#0d0f14);color:#e9edf2;min-height:100vh}
.wrap{max-width:1000px;margin:0 auto;padding:34px 20px 60px}
h1{font-size:26px;margin:0 0 2px} .sub{color:#9aa4b2;margin:0 0 22px;font-size:14px}
.card{background:#161a22;border:1px solid #242b36;border-radius:16px;padding:22px;margin-bottom:18px;box-shadow:0 6px 24px rgba(0,0,0,.25)}
.up{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
input[type=file]{color:#cbd2db;font-size:14px}
button{background:linear-gradient(90deg,#2d7ef7,#4aa8ff);color:#fff;border:0;padding:12px 24px;border-radius:10px;font-size:15px;font-weight:600;cursor:pointer}
.lab{color:#8b95a4;font-size:11px;text-transform:uppercase;letter-spacing:.8px;font-weight:600}
.big{font-size:60px;font-weight:800;color:#4fe08a;margin:4px 0;line-height:1}
.conf{font-size:15px;color:#c3cbd6;margin-top:4px} .kg{font-size:18px;color:#aeb7c4;margin-top:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px} @media(max-width:720px){.grid{grid-template-columns:1fr}}
.gauge{height:20px;background:#0d0f14;border:1px solid #2a323f;border-radius:12px;overflow:hidden;margin:8px 0}
.gfill{height:100%;background:linear-gradient(90deg,#2d7ef7,#4fe08a)}
.steps{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}
.step{flex:1;min-width:150px;background:#12161d;border:1px solid #242b36;border-radius:10px;padding:12px}
.step .n{font-weight:700;font-size:13px} .step .t{font-size:12px;color:#b7c0cc;margin-top:3px}
.loc{color:#4fe08a} .gpu{color:#ffb454}
.stat{display:inline-block;margin-right:26px;margin-top:6px} .stat b{font-size:19px;color:#e9edf2}
.badge{display:inline-block;background:#0f2a1a;color:#4fe08a;border:1px solid #1d5033;border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600}
.badgew{display:inline-block;background:#2a1e0f;color:#ffb454;border:1px solid #50401d;border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600}
img{max-width:100%;border-radius:11px;border:1px solid #2a323f;display:block;margin-top:8px}
.note{font-size:12px;color:#7f8896;line-height:1.6;margin-top:8px} code{color:#9ecbff}
"""

PIPE = """
<div class=card><div class=lab>Pipeline — the whole connectivity</div>
 <div class=steps>
  <div class=step><div class="n loc">1 · Video</div><div class=t>slow orbit of the kiln (phone)</div></div>
  <div class=step><div class="n loc">2 · Frames — LOCAL</div><div class=t>sharp frames extracted on your PC</div></div>
  <div class=step><div class="n gpu">3 · 3-D reconstruct — GPU (Colab)</div><div class=t>frames → dense point cloud (needs a GPU)</div></div>
  <div class=step><div class="n loc">4 · Volume — LOCAL</div><div class=t>this dashboard, on your PC</div></div>
 </div>
 <p class=note>Only step 3 needs a GPU (this machine has none) — everything else runs locally.
 The volume engine here is the same code validated to ~2–4% on ground-truth kilns.</p></div>
"""

PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Kon-Tiki Biochar Volume — Video Dashboard</title><style>{css}</style></head><body><div class=wrap>
<h1>🔥 Kon-Tiki Biochar Volume — Video Dashboard</h1>
<p class=sub>Local dashboard. Upload the 3-D cloud (<code>dense.ply</code>) for the volume, or a video to do the local frame step.</p>
<div class=card><form method=post action=/analyze enctype=multipart/form-data class=up>
 <input type=file name=f accept=".ply,.mp4,.mov,.avi,.mkv" required>
 <button type=submit>Analyse</button>
 <span class=note>.ply → full volume result · video → frame extraction + hand-off</span>
</form></div>
{result}
{pipe}
</div></body></html>"""


def render(result=""):
    return PAGE.format(css=CSS, result=result, pipe=PIPE)


@app.route("/")
def index():
    return render()


@app.route("/analyze", methods=["POST"])
def analyze():
    fs = request.files["f"]; name = (fs.filename or "").lower()
    path = os.path.join(WORK, "upload" + os.path.splitext(name)[1])
    fs.save(path)

    if name.endswith(".ply"):
        try:
            P = load_ply(path)
        except Exception as e:
            return render(f"<div class=card>Could not read that .ply ({e}).</div>")
        res, views, Vfull = analyse_ply(P)
        V = res["volume_L"]; Vf = res["volume_L_flatfill"]
        gap = abs(V - Vf) / max(V, 1) * 100
        band = max(gap / 100 * V, 20)
        fill = max(0, min(100, res["fill_pct"]))
        ok = gap < 8
        trust = (f"<span class=badge>✓ Consistent — estimates agree ({gap:.0f}%)</span>" if ok
                 else f"<span class=badgew>⚠ Estimates disagree {gap:.0f}% — cloud noisy, re-shoot per SOP</span>")
        rimtxt = "≈ correct (~75 cm)" if 0.7 < res["scale_cm_per_unit"] else ""
        result = f"""<div class=card><div class=grid>
          <div>
            <div class=lab>Estimated biochar volume</div>
            <div class=big>{V:,.0f} L</div>
            <div class=conf>± {band:,.0f} L &nbsp; {trust}</div>
            <div class=kg>≈ {V*DENSITY:,.0f} kg · {V/1000:.2f} m³</div>
            <div class=lab style=margin-top:16px>Fill (of ~{Vfull:.0f} L kiln)</div>
            <div class=gauge><div class=gfill style=width:{fill:.0f}%></div></div>
            <div class=note>{fill:.0f}% full · fill height {res['fill_height_cm']:.0f} cm</div>
          </div>
          <div>
            <div class=lab>3-D reconstruction (oriented &amp; scaled)</div>
            <img src="data:image/png;base64,{views}">
            <div class=note>TOP should be a disk (rim); SIDE a cone/bowl.</div>
          </div>
        </div></div>
        <div class=card><div class=lab>Evidence</div><div style=margin-top:8px>
          <span class=stat><span class=lab>Integrated</span><br><b>{V:,.0f} L</b></span>
          <span class=stat><span class=lab>Flat-fill check</span><br><b>{Vf:,.0f} L</b></span>
          <span class=stat><span class=lab>Fill height</span><br><b>{res['fill_height_cm']:.0f} cm</b></span>
          <span class=stat><span class=lab>Recovered scale</span><br><b>{res['scale_cm_per_unit']:.3f} cm/u</b></span>
          <span class=stat><span class=lab>Cloud points</span><br><b>{len(P):,}</b></span>
        </div><p class=note>Rim assumed Ø150 cm (Kon-Tiki 1000). Volume integrates the measured
        biochar surface against the known cone; the flat-fill number is a cross-check.</p></div>"""
        return render(result)

    # else: a video -> local frame extraction + hand-off
    total, montage = extract_frames_montage(path)
    if not montage:
        return render("<div class=card>Could not read that video.</div>")
    result = f"""<div class=card><div class=lab>Step 2 · Frames extracted — LOCAL ✓</div>
      <div class=note>{total} video frames scanned. Sample below (evenly spaced).</div>
      <img src="data:image/jpeg;base64,{montage}">
      <div style=margin-top:14px><span class=badgew>Next: Step 3 needs a GPU</span></div>
      <p class=note>Reconstruction (frames → 3-D point cloud) runs on the Colab GPU notebook
      (<code>KonTiki_Video_to_Volume.ipynb</code>). It saves <code>dense.ply</code> — download it and
      upload it here to get the volume, all locally.</p></div>"""
    return render(result)


if __name__ == "__main__":
    print("Open http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
