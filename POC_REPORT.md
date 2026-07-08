# POC Report — Vision-Based Volume & Weight Estimation of Granular Charcoal

**Phase:** Software-only feasibility (no camera hardware).
**Question:** Can a software pipeline estimate the **volume (L)** and **weight (kg)**
of loose granular charcoal in a cylindrical container from a single 2-D image, to a
measurable accuracy?

**Answer (this POC): yes for the volume/weight math and the segmentation, with the
heaped surface as the dominant error term — quantified below.**

---

## 0. In plain English (read this first)

**The problem.** We want to know *how much charcoal is in a barrel* — its volume
(litres) and weight (kg) — using **just a camera photo**, with no scale, no manual
dipping, no opening the lid. Think of an automatic "fuel gauge" for a bin of charcoal.

**The idea, in one line.** Take a photo → an AI traces the charcoal surface in the
image → because we know the barrel's shape and the camera's position, we convert the
surface height into a volume → multiply by charcoal density to get weight.

**This phase has no hardware.** We prove the idea in *software only*, using two sets of
images: realistic 3-D renders where the true answer is known exactly (to measure
accuracy), and real photos from a public research dataset (to prove it survives messy,
real-world images). The camera rig itself is a written recommendation for the next phase.

![how it works](../results/figures/explainer.png)

**Bottom line:** on images where we know the true answer, the system was within
**~3%** on both volume and weight, and the AI correctly traced the contents in
**~98%** of the pixels (synthetic) and **~90%** on real, unseen containers — meeting
every accuracy target we set. Details and the honest caveats are below.

> **Glossary (plain terms):** *Segmentation* = the AI outlining which pixels are the
> charcoal/container. *Mask* = that outline as a black-and-white stencil. *IoU* = how
> well the AI's outline overlaps the true outline (1.0 = perfect, ≥0.9 = excellent).
> *MAPE* = average percentage error (lower = better). *Track A* = real photos,
> *Track B* = synthetic renders. *Ground truth* = the known correct answer.

---

## 1. What was actually built and run

A complete, runnable pipeline plus two datasets with ground truth:

1. **Track B — synthetic (Blender).** 400 photorealistic renders of an opaque,
   open-top cylinder partially filled with a dark, heaped granular material, viewed
   from an elevated oblique camera. For every image we know the radius, fill height,
   **exact volume (litres) and weight (kg)**, pixel-perfect masks, and the full camera
   calibration. This is the only place absolute accuracy can be measured, so it
   carries the headline number.
2. **Track A — real (CORSMAL C-CCM).** A respected academic benchmark of real RGB
   frames with rice/pasta (granular charcoal proxies). Used to prove the segmenter
   survives real, cluttered images. C-CCM has **no capacity/mass ground truth**, so
   only segmentation IoU and relative fill-level are scoreable here — not litres/kg.

Both feed one shared pipeline: **segment → fill-height → volume → weight → score.**

---

## 2. Datasets

### 2.1 Track B — synthetic (generated here)
- **400 images**, 640×640, Cycles + OptiX GPU, domain-randomised lighting, camera
  pose, container size/material, and granular form (lump vs briquette).
- Container radius 9–20 cm, fill height 6–40 cm; bounded freeboard + adaptive camera
  elevation so the charcoal surface always clears the near rim.
- **Ground truth per image:** `radius, height, fill_height, angle-of-repose, V_cyl,
  V_cone, V_analytic, V_mesh (authoritative), bulk_density, weight, camera K/R/t.`
- The authoritative volume is `bmesh.calc_volume()` of the actual rendered fill mesh;
  it agrees with the analytic cylinder+cone formula to **<0.15 %**, confirming the
  geometry and ground truth are internally consistent.
- Outputs: `rgb/`, class-colour `mask/`, binary `fill_mask/` & `drum_mask/`, QA
  overlays, and `manifest.json`. ArUco scale marker present in every scene.

### 2.2 Track A — real (CORSMAL C-CCM)
- Source: Zenodo record 4642577 (the QMUL hosting site was unreachable; the still-
  image variant on Zenodo is exactly what the pipeline needs).
- Held locally: all 10,216 container masks + annotations, and a **1,217-image RGB
  shard** (ids 9000–10216) of which **660 are granular** (rice/pasta), spanning all
  8 containers and 3 fill levels.
- Container-disjoint split (evaluate on **unseen** containers): train {1,2,4,5},
  val {3,6}, test {7,8}.
- **Honest limitation:** masks are *container* masks (soft Mask R-CNN maps, binarised
  at 128), not fill-region masks; labels are categorical (type + level), with no
  capacity → no absolute volume/weight.

---

## 3. Method

### 3.1 Stage 1 — Segmentation (≥2 methods compared)
- **Classical CV (no training):** bright-background removal → Otsu on the object
  interior to isolate the dark charcoal → largest connected component. Control baseline.
- **YOLO11n-seg (learned):** trained on each track's masks. Recommended method.

### 3.2 Stage 2 — Fill height from the mask
Because the synthetic container is a known vertical cylinder (radius r, base z=0) with
exact camera calibration, we **back-project** the top boundary of the charcoal mask
and intersect each ray with the cylinder wall `x²+y²=r²` to read world height. We take
the topmost charcoal pixel per column (requiring a short vertical run, to reject
speckle) over a central band, and use the median. A **2-parameter linear calibration**
`h = a·h_raw + b` is fit on the **train** split and applied on test — this is the
"calibrate heap/parallax offset on synthetic ground truth" step; no per-image GT is
used at test time. With GT masks this estimator is unbiased (a≈1.0, b≈0, ≤1 mm error).

### 3.3 Stage 3-4 — Volume & weight
`V = π r² h + ⅓ π r³ tan θ` (θ = 38° nominal heaped cone), `W = ρ_bulk · V`.
Bulk density is taken from the (known) synthetic value so weight error isolates the
pipeline rather than density guesswork — exactly as the brief specifies.

### 3.4 Stage 5 — Scoring
Volume/weight **MAPE**, mask **IoU**, height MAE; broken down by fill level and
material form. Track A scores container IoU on unseen containers.

---

## 4. Results

### 4.0 Success-criteria scorecard

| Metric | POC target | Achieved (best method) | Verdict |
|---|---|---|---|
| Volume error, cylinder body | ≤ ±5% | **2.7%** (YOLO, +heap) | ✅ |
| Volume error, incl. heaped top | ≤ ±10% | **2.7%** | ✅ |
| Weight error | ≤ ±10–15% | **2.7%** | ✅ |
| Fill-region IoU (synthetic) | ≥ 0.90 | **0.984** (YOLO) | ✅ |
| Container IoU (real, unseen) | ≥ 0.90 | **0.901** (YOLO) | ✅ |

All five POC targets are met. The decisive levers are (a) a learned segmenter —
classical CV fails (IoU 0.50 synthetic / 0.39 real) — and (b) the heaped-cone
correction, which alone moves volume error from 16.5% (cylinder-only) to 2.7%.

### 4.1 Track B — synthetic, absolute accuracy (held-out test split)

| Segmentation | fill IoU | height MAE | volume MAPE (cyl) | volume MAPE (+heap) | weight MAPE |
|---|---|---|---|---|---|
| GT mask (geometry ceiling) | 1.000 | 0.9 mm | 16.5% | **2.7%** | 2.7% |
| YOLO11-seg (learned) | 0.984 | 1.8 mm | 16.5% | **2.7%** | 2.7% |
| Classical CV (baseline) | 0.495 | 38.5 mm | 24.2% | **16.6%** | 16.6% |

**Breakdown (YOLO11-seg (learned)), heaped volume MAPE:**

| condition | n | volume MAPE | IoU |
|---|---|---|---|
| form=briquette | 31 | 3.3% | 0.985 |
| form=lump | 29 | 2.1% | 0.983 |
| fill=high | 45 | 2.0% | 0.983 |
| fill=low | 15 | 4.7% | 0.987 |

_Test split: 60 held-out images. Calibration fit on the train split only._


### 4.2 Track A — real CORSMAL C-CCM, container segmentation (unseen containers)

| Method | container IoU (mean) | median | frac > 0.5 | frac > 0.9 |
|---|---|---|---|---|
| Classical CV (no training) | 0.388 | 0.400 | 0.31 | – |
| YOLO11n-seg (learned) | **0.901** | 0.925 | 1.00 | 0.68 |

_Granular (rice/pasta) frames — train 428, val 145, test 87; test = unseen containers [7, 8]._


_No capacity/mass GT in C-CCM, so absolute volume/weight are not scoreable on the real track — they come from Track B (§4.1)._


![volume](../results/figures/vol_pred_vs_true.png)
![methods](../results/figures/method_comparison.png)
![tracka](../results/figures/track_a_iou.png)

---

## 5. Limitations (unchanged from the brief, confirmed in practice)

- **Proxy materials:** rice/pasta are *light*, not near-black like charcoal — the
  classical "dark content" heuristic does not transfer to the real proxies (observed
  directly), which is itself evidence that a learned segmenter is required.
- **Single-view ceiling:** a 2-D view cannot see the centre/back of the 3-D heap; the
  heaped cone is the dominant volume-error term and motivates a load cell / second view.
- **No absolute GT on real data:** C-CCM lacks capacity/mass; absolute litres/kg are
  validated only synthetically.
- **Synthetic-to-real gap:** Blender realism omits dust, glare and sensor noise.

---

## 6. Phase-2 hardware proposal (recommendation only)

Fixed fronto-parallel (or elevated oblique) camera; controlled diffuse LED enclosure;
cross-polariser for glossy briquettes; ArUco fiducials in the fill plane; **load cell
under the vessel** to validate weight and auto-calibrate density; IP67 + air-purge
optics; Jetson Orin Nano with YOLO-seg → TensorRT FP16. Decision thresholds: if
side-view volume error > ±5 % after calibration → add a second/depth camera; if weight
error dominates → lean on the load cell; if contrast is poor → invest in active
illumination first.

---

## 7. Where to mount the camera — digital-twin angle study

A direct, money-saving use of the synthetic engine: build the target drum as a
**digital twin** and find the best camera position **before** any hardware is bought.
We fixed one drum (radius 15 cm, height 50 cm; lump charcoal, 40° pile) and rendered
the *same known fill* from 7 camera elevations × 2 directions × 3 fill levels
(42 views), then ran the pipeline on each.

![angle montage](../results/figures/angle_montage.png)

![angle sweep](../results/figures/angle_sweep.png)

**Findings (for this drum):**

| Camera elevation | What happens | Volume error |
|---|---|---|
| 20–30° (side-on) | near rim **hides** the charcoal — system goes blind | fails / ~13% |
| **45–70° (looking down in)** | **full surface visible — sweet spot** | **~0.6–1%** |
| 80° (near top-down) | works, but loses the wall fill-line on taller/narrower drums | ~0.7–2% |

**Recommendation:** mount the camera **above the opening, angled down ~50–60°**.
The exact best angle depends on the drum's height-to-width ratio and the pile steepness,
so for a real container we re-run this sweep on its true measurements and return a
specific "mount here, expect ±X%" answer. To retarget: edit the `CONTAINER` / `CHARCOAL`
constants at the top of `synthetic/scripts/angle_sweep.py`, then
`blender -b --python angle_sweep.py` and `python3 src/analyze_sweep.py`.
