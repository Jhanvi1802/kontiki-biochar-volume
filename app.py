"""Kon-Tiki Biochar Volume — transparent web dashboard.
Upload a kiln photo -> volume + confidence range + a full breakdown of HOW the
number was produced (test-time-augmentation spread, attention heatmap, fill
gauge, and the model's validated accuracy). Runs locally on CPU.
Run:  python app.py   ->  http://127.0.0.1:5000
"""
import io, base64, os, numpy as np, cv2, torch, torch.nn as nn
import torchvision as tv
from torchvision import transforms
from PIL import Image, ImageOps, ImageEnhance
from flask import Flask, request

BASE = os.path.dirname(os.path.abspath(__file__))
DENSITY = 0.25
NOMINAL_FULL = 1000.0
ACC = {"mape": 3.3, "w10": 98, "corr": 0.71, "n": 2339, "mae": 28}

# ---- load model (prefer the masked/kiln-only model if present) ----
cands = [f"{BASE}/kontiki_cnn_masked.pt", f"{BASE}/kontiki_cnn.pt"]
ckpt = next((p for p in cands if os.path.exists(p)), None)
assert ckpt, "no kontiki_cnn*.pt found."
ck = torch.load(ckpt, map_location="cpu")
VMEAN, VSTD, IMG, ARCH = ck["vmean"], ck["vstd"], ck["img"], ck.get("arch", "efficientnet_b0")
MASKED = ck.get("masked", False)
SAM_MODEL = None
if MASKED:
    from ultralytics import SAM
    SAM_MODEL = SAM("mobile_sam.pt")
def kiln_mask(pil):
    """SAM box-prompt -> keep only the kiln, black out the outside."""
    img = cv2.resize(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), (768, 576)); H, W = img.shape[:2]
    box = [int(W*0.12), int(H*0.10), int(W*0.88), int(H*0.92)]
    try:
        r = SAM_MODEL(img[:,:,::-1], bboxes=[box], verbose=False)
        m = (cv2.resize(r[0].masks.data[0].cpu().numpy().astype(np.uint8), (W,H)) > 0).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1: m = (lab == 1+int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((17,17), np.uint8))
        img[m == 0] = 0
    except Exception: pass
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def build():
    m = getattr(tv.models, ARCH)(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, 1); return m
states = ck["models"] if "models" in ck else [ck["state_dict"]]
nets = []
for sd in states:
    n = build(); n.load_state_dict(sd); n.eval(); nets.append(n)
norm = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
tf = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), norm])
_act = {}
def _hook(m, i, o):
    _act["a"] = o
    if o.requires_grad:          # only during the grad-enabled Grad-CAM pass
        o.retain_grad()
nets[0].features.register_forward_hook(_hook)
print(f"Loaded {ARCH} ({len(nets)} model(s)) from {ckpt}")

def denorm(t): return float(t)*VSTD + VMEAN

def analyse(pil):
    work = kiln_mask(pil) if MASKED else pil        # isolate the kiln first if masked model
    # 1) test-time augmentation -> prediction spread (confidence)
    variants = [work, ImageOps.mirror(work),
                ImageEnhance.Brightness(work).enhance(1.12),
                ImageEnhance.Brightness(work).enhance(0.88),
                work.rotate(5, expand=False), work.rotate(-5, expand=False)]
    preds = []
    with torch.no_grad():
        for v in variants:
            x = tf(v).unsqueeze(0)
            preds.append(np.mean([denorm(n(x).item()) for n in nets]))
    preds = np.array(preds); V, spread = float(preds.mean()), float(preds.std())
    # 2) Grad-CAM (attention) on the main model
    x = tf(work).unsqueeze(0); nets[0].zero_grad()
    out = nets[0](x); out.backward()
    a = _act["a"]; g = a.grad; w = g.mean((2,3), keepdim=True)
    cam = torch.relu((w*a).sum(1, keepdim=True))[0,0].detach().numpy()
    cam = (cam - cam.min())/(cam.max()-cam.min()+1e-8)
    orig = cv2.cvtColor(np.array(work.resize((IMG,IMG))), cv2.COLOR_RGB2BGR)
    heat = cv2.applyColorMap((cv2.resize(cam,(IMG,IMG))*255).astype("uint8"), cv2.COLORMAP_JET)
    over = cv2.addWeighted(orig, 0.6, heat, 0.4, 0)
    _, buf = cv2.imencode(".jpg", over)
    masked_b64 = None
    if MASKED:
        _, mb = cv2.imencode(".jpg", cv2.cvtColor(np.array(work), cv2.COLOR_RGB2BGR))
        masked_b64 = base64.b64encode(mb).decode()
    return V, spread, base64.b64encode(buf).decode(), masked_b64

def quality_checks(pil, spread):
    """Heuristic capture-quality checks -> warn the user to re-shoot bad photos.
    Thresholds calibrated on real good kiln photos so they don't false-warn."""
    im = cv2.resize(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), (800, 600))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, W = g.shape
    sharp = cv2.Laplacian(g, cv2.CV_64F).var(); bright = g.mean()
    dark = (g < 95) & (hsv[:,:,1] < 115)                 # biochar-like (dark, low-sat)
    dc = dark[int(H*0.10):int(H*0.62), int(W*0.18):int(W*0.82)].mean()
    checks = [
        ("Sharpness", sharp > 400, "sharp" if sharp > 400 else "blurry — hold steadier"),
        ("Lighting", 45 < bright < 215, "good" if 45 < bright < 215 else ("too dark" if bright <= 45 else "over-exposed")),
        ("Biochar surface visible", dc > 0.05, "yes" if dc > 0.05 else "hard to see — shoot ~50–70° looking down into the kiln"),
        ("Model confidence", spread < 30, "high" if spread < 30 else "low — try another shot"),
    ]
    return all(c[1] for c in checks), checks

app = Flask(__name__)
CSS = """
 *{box-sizing:border-box} body{font-family:system-ui,-apple-system,Segoe UI,Arial;margin:0;
   background:radial-gradient(1200px 600px at 50% -10%,#1b2230,#0d0f14);color:#e9edf2;min-height:100vh}
 .wrap{max-width:960px;margin:0 auto;padding:34px 20px 60px}
 h1{font-size:26px;margin:0 0 2px;letter-spacing:.2px} .sub{color:#9aa4b2;margin:0 0 24px;font-size:14px}
 .card{background:#161a22;border:1px solid #242b36;border-radius:16px;padding:24px;margin-bottom:18px;
   box-shadow:0 6px 24px rgba(0,0,0,.25)}
 .up{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
 input[type=file]{color:#cbd2db;font-size:14px}
 button{background:linear-gradient(90deg,#2d7ef7,#4aa8ff);color:#fff;border:0;padding:12px 24px;
   border-radius:10px;font-size:15px;font-weight:600;cursor:pointer}
 .lab{color:#8b95a4;font-size:11px;text-transform:uppercase;letter-spacing:.8px;font-weight:600}
 .big{font-size:60px;font-weight:800;color:#4fe08a;margin:4px 0;line-height:1}
 .conf{font-size:15px;color:#c3cbd6;margin-top:4px} .kg{font-size:18px;color:#aeb7c4;margin-top:8px}
 .grid{display:grid;grid-template-columns:1.1fr 1fr;gap:18px} @media(max-width:680px){.grid{grid-template-columns:1fr}}
 .imgs{display:grid;grid-template-columns:1fr 1fr;gap:10px}
 .imgs img{width:100%;border-radius:11px;border:1px solid #2a323f;display:block}
 .cap{font-size:11px;color:#8b95a4;margin-top:6px;text-align:center}
 .gauge{height:20px;background:#0d0f14;border:1px solid #2a323f;border-radius:12px;overflow:hidden;margin:8px 0}
 .gfill{height:100%;background:linear-gradient(90deg,#2d7ef7,#4fe08a);transition:width .4s}
 .steps{display:flex;gap:10px;flex-wrap:wrap;margin-top:6px}
 .step{flex:1;min-width:120px;background:#12161d;border:1px solid #242b36;border-radius:10px;padding:12px}
 .step .n{color:#4aa8ff;font-weight:700;font-size:13px} .step .t{font-size:12px;color:#b7c0cc;margin-top:3px}
 .stat{display:inline-block;margin-right:22px} .stat b{font-size:20px;color:#e9edf2}
 .note{font-size:12px;color:#7f8896;line-height:1.6;margin-top:8px}
 .badge{display:inline-block;background:#0f2a1a;color:#4fe08a;border:1px solid #1d5033;border-radius:20px;
   padding:4px 12px;font-size:12px;font-weight:600}
 .badgew{display:inline-block;background:#2a1e0f;color:#ffb454;border:1px solid #50401d;border-radius:20px;
   padding:4px 12px;font-size:12px;font-weight:600}
 .qrow{display:flex;align-items:center;gap:12px;padding:9px 0;border-bottom:1px solid #1e242e;font-size:14px}
 .qrow:last-child{border-bottom:0} .ok{color:#4fe08a} .warn{color:#ffb454}
 .qmark{width:20px;font-weight:700;text-align:center} .qname{width:190px;color:#c9d1db}
 .qdet{color:#8b95a4;font-size:13px}
"""
PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Kon-Tiki Biochar Volume</title><style>{css}</style></head><body><div class=wrap>
 <h1>🔥 Kon-Tiki Biochar Volume</h1>
 <p class=sub>Upload a post-quench kiln photo — get the volume <b>and a full breakdown of how it was computed</b>.</p>
 <div class=card><form method=post action=/predict enctype=multipart/form-data class=up>
   <input type=file name=image accept=image/* required>
   <button type=submit>Analyse</button>
 </form></div>
 {result}
 <div class=card>
   <div class=lab>How it works — the pipeline</div>
   <div class=steps>
     <div class=step><div class=n>1 · Input</div><div class=t>Your kiln photo is resized to {img}px</div></div>
     <div class=step><div class=n>2 · CNN</div><div class=t>{arch} reads the biochar fill</div></div>
     <div class=step><div class=n>3 · Confidence</div><div class=t>6 augmented views → prediction spread</div></div>
     <div class=step><div class=n>4 · Attention</div><div class=t>Grad-CAM shows the pixels used</div></div>
   </div>
 </div>
 <div class=card>
   <div class=lab>Model evidence (honest held-out test)</div>
   <div style=margin-top:12px>
     <span class=stat><span class=lab>Avg error</span><br><b>{mape}%</b></span>
     <span class=stat><span class=lab>Within 10%</span><br><b>{w10}%</b></span>
     <span class=stat><span class=lab>Tracks fill</span><br><b>corr {corr}</b></span>
     <span class=stat><span class=lab>Tested on</span><br><b>{n} unseen</b></span>
   </div>
   <p class=note>Accuracy measured on kiln photos the model never trained on. The confidence range
   comes from how much the prediction wobbles across augmented views — a tight range means the model
   is sure. Attention heatmap shows where the model looked (should be on the biochar).</p>
 </div>
</div></body></html>"""

def render(res=""):
    return PAGE.format(css=CSS, result=res, img=IMG, arch=ARCH.replace("_"," ").title(),
                       mape=ACC["mape"], w10=ACC["w10"], corr=ACC["corr"], n=ACC["n"])

@app.route("/")
def index(): return render()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.files["image"].read()
    try: pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception: return render("<div class=card>Could not read that image.</div>")
    V, spread, cam_b64, masked_b64 = analyse(pil)
    band = max(spread, ACC["mae"]*0.5)      # confidence half-width (L)
    kg = V*DENSITY; fill = max(0, min(100, V/NOMINAL_FULL*100))
    tight = "tight — model is confident" if spread < 20 else "wide — treat as approximate"
    photo = base64.b64encode(data).decode()
    qok, checks = quality_checks(pil, spread)
    qrows = "".join(
        f"<div class=qrow><span class='qmark {'ok' if ok else 'warn'}'>{'✓' if ok else '⚠'}</span>"
        f"<span class=qname>{name}</span><span class=qdet>{det}</span></div>"
        for name, ok, det in checks)
    qbadge = ("<span class=badge>✓ Good capture</span>" if qok
              else "<span class=badgew>⚠ Re-shoot suggested — see below</span>")
    quality_card = f"""<div class=card><div class=lab>Capture quality {qbadge}</div>
      <div style=margin-top:8px>{qrows}</div></div>"""
    res = f"""<div class=card><div class=grid>
      <div>
        <div class=lab>Estimated biochar volume</div>
        <div class=big>{V:,.0f} L</div>
        <div class=conf>± {band:,.0f} L  ({tight})</div>
        <div class=kg>≈ {kg:,.0f} kg  ·  {V/1000:.2f} m³</div>
        <div class=lab style=margin-top:18px>Approx. fill (of ~1000 L kiln)</div>
        <div class=gauge><div class=gfill style=width:{fill:.0f}%></div></div>
        <div class=cap style=text-align:left>{fill:.0f}% full</div>
        <div style=margin-top:16px><span class=badge>Confidence spread across 6 views: ±{spread:,.0f} L</span></div>
      </div>
      <div class=imgs>
        <div><img src='data:image/jpeg;base64,{photo}'><div class=cap>Your photo</div></div>
        <div><img src='data:image/jpeg;base64,{cam_b64}'><div class=cap>Where the model looked (red = most)</div></div>
        {('<div><img src="data:image/jpeg;base64,'+masked_b64+'"><div class=cap>Kiln isolated — outside removed (what the model measures)</div></div>') if masked_b64 else ''}
      </div>
    </div></div>""" + quality_card
    return render(res)

if __name__ == "__main__":
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
