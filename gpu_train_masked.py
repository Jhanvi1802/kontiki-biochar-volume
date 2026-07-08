# =============================================================================
# Kon-Tiki Biochar Volume — MASKED training (Colab GPU)
# Removes the OUTSIDE material: SAM masks each photo to the kiln only, then the
# CNN trains on kiln-only images (can't use the background). Fixes the shortcut.
# Cell 1: upload Excel.  Cell 2: run (installs SAM, masks, trains, evaluates).
# Runtime > Change runtime type > GPU (T4).
# =============================================================================


# %% CELL 1 — upload Excel
from google.colab import files
import glob
print("Upload your Excel (image links + manual volumes)...")
files.upload(); EXCEL = glob.glob("*.xlsx")[0]; print("Using:", EXCEL)


# %% CELL 2 — install SAM, mask every image, train on kiln-only images
import subprocess, sys
subprocess.run("pip -q install ultralytics", shell=True)
import os, numpy as np, torch, torch.nn as nn, requests, random, openpyxl, cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import torchvision as tv
from torchvision import transforms
from ultralytics import SAM
import matplotlib.pyplot as plt

assert torch.cuda.is_available(), "Enable GPU"
dev = "cuda"
N_IMAGES, EPOCHS, IMG, BATCH = 8000, 20, 256, 32     # 8k masked images is plenty

# ---- read Excel ----
ws = openpyxl.load_workbook(EXCEL)["Correct"]; rows = []
for r in range(2, ws.max_row+1):
    idc = ws.cell(r,1).value; link = ws.cell(r,2); man = ws.cell(r,4).value
    if idc and link.hyperlink and man is not None:
        rows.append((int(idc), float(man), link.hyperlink.target))
random.seed(0); random.shuffle(rows); rows = rows[:N_IMAGES]
print(len(rows), "rows")

# ---- download ----
os.makedirs("/content/raw", exist_ok=True)
def dl(row):
    idc, vol, url = row; fn = f"/content/raw/{idc}.jpg"
    if os.path.exists(fn) and os.path.getsize(fn) > 5000: return (fn, vol)
    try:
        rr = requests.get(url, timeout=25)
        if rr.status_code == 200 and len(rr.content) > 5000:
            open(fn,"wb").write(rr.content); return (fn, vol)
    except Exception: return None
with ThreadPoolExecutor(max_workers=32) as ex:
    data = [x for x in ex.map(dl, rows) if x]
print("downloaded", len(data))

# ---- SAM: mask each image to the kiln (box prompt + cleanup), cache ----
sam = SAM("mobile_sam.pt")
os.makedirs("/content/masked", exist_ok=True)
def clean(m):
    m = (m > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((17,17), np.uint8))
def mask_kiln(fn):
    out = f"/content/masked/{os.path.basename(fn)}"
    if os.path.exists(out): return out
    img = cv2.imread(fn)
    if img is None: return None
    img = cv2.resize(img, (768, 576)); H, W = img.shape[:2]
    box = [int(W*0.12), int(H*0.10), int(W*0.88), int(H*0.92)]
    try:
        r = sam(img[:,:,::-1], bboxes=[box], verbose=False)
        m = clean(cv2.resize(r[0].masks.data[0].cpu().numpy().astype(np.uint8), (W,H)))
    except Exception:
        m = np.ones((H,W), np.uint8)
    out_img = img.copy(); out_img[m == 0] = 0        # black out everything outside the kiln
    cv2.imwrite(out, out_img); return out
print("masking with SAM (a while on first run)...")
mdata = []
for i,(fn,vol) in enumerate(data):
    mo = mask_kiln(fn)
    if mo: mdata.append((mo, vol))
    if i % 500 == 0: print(f"  masked {i}/{len(data)}")
print("masked", len(mdata))

# ---- train CNN on masked (kiln-only) images ----
random.shuffle(mdata)
vols = np.array([v for _,v in mdata]); VMEAN, VSTD = float(vols.mean()), float(vols.std())
n = len(mdata); nte = int(n*0.12); nval = int(n*0.12)
test, val, train = mdata[:nte], mdata[nte:nte+nval], mdata[nte+nval:]
norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
tf_tr = transforms.Compose([transforms.Resize((IMG,IMG)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15,0.15,0.15), transforms.ToTensor(), norm])
tf_ev = transforms.Compose([transforms.Resize((IMG,IMG)), transforms.ToTensor(), norm])
class DS(torch.utils.data.Dataset):
    def __init__(s, it, tf): s.it, s.tf = it, tf
    def __len__(s): return len(s.it)
    def __getitem__(s, i):
        fn, v = s.it[i]
        return s.tf(Image.open(fn).convert("RGB")), torch.tensor([(v-VMEAN)/VSTD], dtype=torch.float32)
L = lambda it, tf, sh: torch.utils.data.DataLoader(DS(it,tf), batch_size=BATCH, shuffle=sh, num_workers=2)
dl_tr, dl_va, dl_te = L(train,tf_tr,True), L(val,tf_ev,False), L(test,tf_ev,False)

net = tv.models.efficientnet_b0(weights=tv.models.EfficientNet_B0_Weights.DEFAULT)
net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1); net = net.to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); lossf = nn.SmoothL1Loss()
def ev(loader):
    net.eval(); P,Y=[],[]
    with torch.no_grad():
        for x,y in loader:
            P += list(net(x.to(dev)).cpu().numpy().ravel()*VSTD+VMEAN); Y += list(y.numpy().ravel()*VSTD+VMEAN)
    P,Y=np.array(P),np.array(Y); return np.mean(np.abs(P-Y)), np.mean(np.abs(P-Y)/Y)*100, P, Y
best=1e9
for e in range(EPOCHS):
    net.train()
    for x,y in dl_tr:
        opt.zero_grad(); lossf(net(x.to(dev)), y.to(dev)).backward(); opt.step()
    sched.step(); mae,mape,_,_=ev(dl_va)
    print(f"epoch {e+1}/{EPOCHS} val MAE={mae:.1f} MAPE={mape:.1f}%")
    if mae<best: best=mae; torch.save(net.state_dict(), "/content/best.pt")
net.load_state_dict(torch.load("/content/best.pt"))
mae,mape,P,Y=ev(dl_te); corr=np.corrcoef(P,Y)[0,1]; slope=np.polyfit(Y,P,1)[0]
w10=np.mean(np.abs(P-Y)/Y<.1)*100
print(f"\n=== MASKED held-out (n={len(Y)}) MAE={mae:.1f} MAPE={mape:.1f}% corr={corr:.2f} slope={slope:.2f} w10={w10:.0f}% ===")
plt.figure(figsize=(6,6)); plt.scatter(Y,P,s=12,alpha=.4)
lo,hi=min(Y.min(),P.min()),max(Y.max(),P.max()); plt.plot([lo,hi],[lo,hi],'r--')
plt.xlabel("TRUE L"); plt.ylabel("PRED L"); plt.title(f"Masked CNN: MAPE={mape:.1f}% corr={corr:.2f}")
plt.grid(alpha=.3); plt.savefig("/content/masked_scatter.png",dpi=120); plt.show()
torch.save({"state_dict":net.state_dict(),"vmean":VMEAN,"vstd":VSTD,"img":IMG,
            "arch":"efficientnet_b0","masked":True}, "/content/kontiki_cnn_masked.pt")
files.download("/content/kontiki_cnn_masked.pt"); files.download("/content/masked_scatter.png")
print("Done. Send kontiki_cnn_masked.pt back to plug into the web app.")
