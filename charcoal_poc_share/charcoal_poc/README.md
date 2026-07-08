# POC — Vision-Based Volume & Weight Estimation of Granular Charcoal in a Cylindrical Container

Software-only feasibility study. **No camera hardware.** Two data tracks feed one
shared estimation pipeline; together they show the method is sound on real images
*and* quantify absolute accuracy on the exact target geometry.

```
data/ccm/            Track A — real benchmark (CORSMAL C-CCM)  [downloaded]
synthetic/           Track B — Blender-rendered cylinders        [generated here]
src/                 pipeline: segmentation, geometry, evaluation
results/             metrics JSON + figures
docs/POC_REPORT.md   the write-up with results
tools/               portable Blender 4.2.5 (headless, OptiX GPU)
```

## The two tracks

| | Track A — CORSMAL C-CCM (real) | Track B — synthetic (Blender) |
|---|---|---|
| Images | 10,216 real RGB frames (we hold a 1,217 shard); 660 granular | 400 rendered cylinders (we generate) |
| Filling | rice / pasta (charcoal proxies) | dark heaped granular (charcoal-like) |
| Geometry | cups & glasses (no cylinder) | true opaque cylinder + heaped top |
| Ground truth | container mask, filling type & **level (0/50/90%)** | **exact** radius, fill height, volume (L), weight (kg), camera calib |
| Proves | segmentation survives real, cluttered images | the volume/weight math is accurate (clean ±% error) |
| **Limitation** | **no capacity/mass GT → no absolute volume/weight** | synthetic-to-real gap |

> **Honest scope.** C-CCM provides *classification* labels, not capacity, so
> absolute volume/weight in litres/kg **cannot** be scored on the real track —
> only segmentation IoU and relative fill level. The absolute accuracy headline
> therefore comes from Track B, where every litre is known exactly. This split is
> by design (see the POC brief).

## Pipeline (shared by both tracks)

`segment → fill height → volume (cylinder + heaped cone) → weight (× bulk density) → score`

- **Stage 1 — Segmentation:** classical CV (no-train baseline) and YOLO11-seg (learned).
- **Stage 2 — Fill height:** back-project the mask's top boundary onto the known
  container cylinder using the exact camera calibration; robust over a central band;
  2-param linear calibration fit on the **train** split (absorbs heap/parallax bias).
- **Stage 3 — Volume:** `π r² h` + heaped cone `⅓π r³ tan θ` (θ = 38° nominal).
- **Stage 4 — Weight:** `ρ_bulk · V` (density known per sample, isolating pipeline error).
- **Stage 5 — Score:** volume/weight MAPE, mask IoU, broken down by condition.

## Reproduce

```bash
# Track B — generate synthetic data (Blender headless, OptiX)
BL=$(cat tools/blender_path.txt)
"$BL" -b --python synthetic/scripts/generate.py -- --n 400 --out synthetic/renders --res 640 --samples 96
python3 src/postprocess_synthetic.py --renders synthetic/renders   # binary masks + QA
python3 src/train_yolo.py --renders synthetic/renders --epochs 60  # Method 1
python3 src/evaluate.py   --renders synthetic/renders              # IoU + vol/weight MAPE

# Track A — real CORSMAL C-CCM (already downloaded to data/ccm)
python3 src/evaluate_ccm.py --root data/ccm --train_yolo --epochs 40
python3 src/make_figures.py
```

See [docs/POC_REPORT.md](docs/POC_REPORT.md) for results and the Phase-2 hardware proposal.
