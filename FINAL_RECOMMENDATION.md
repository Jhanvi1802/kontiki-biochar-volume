# Biochar Volume from a Photo — Final Recommendation (proven method)

## The proven method (PoC: ~2.7% volume & weight error, all 5 targets met)
**Single photo + known container shape — NOT full 3D reconstruction.**
```
1 photo -> AI (YOLO-seg) traces the biochar SURFACE
        -> because the container shape + camera are known, read the FILL LEVEL
        -> volume (cone/frustum + heap) -> weight = density x volume
```
Why this works where multi-view reconstruction struggled: the container shape is KNOWN,
so we only estimate the fill level from one image (no fragile 3D, no height-collapse).
Proof + scorecard + charts: `charcoal_poc_share/charcoal_poc/` (run `demo_cpu.py` ->
2.7% AI ceiling vs 16% no-AI; open `POC_results.ipynb` for slides).

## Camera angle (from the digital-twin angle study / angle_montage.png)
- 20-30 deg (side-on): BLIND, the rim hides the charcoal.
- **45-70 deg (looking down in): sweet spot, 0.5-1.2% error.**
- Recommendation: mount/aim the camera **~50-60 deg above, looking down into the kiln.**

## Applied to the Kon-Tiki 1000 (from design_Kon-Tiki 1000.pdf)
Geometry (FRONT ELEVATION): rim Ø1500 (R=75cm), bottom Ø~803 (r=40cm), depth 930mm (93cm).
Frustum capacity = **998 L** vs the drawing's 1000 L label = **99.8% match** (validates dims).

Calculator: `kiln_volume_calc.py`. Scale-FREE (no marker needed — the kiln IS the ruler):
measure u = (biochar surface circle radius) / (rim circle radius) from the photo, then:

| surf/rim u | fill height | volume | fill % | weight (~0.25 kg/L) |
|---|---|---|---|---|
| 0.70 | 33 cm | 224 L | 22% | 56 kg |
| 0.80 | 53 cm | 423 L | 42% | 106 kg |
| 0.90 | 73 cm | 679 L | 68% | 170 kg |
| 1.00 | 93 cm | 998 L | 100% | 250 kg |

Biochar bulk density ~0.20-0.30 kg/L (confirm by weighing one batch -> calibrates kg).

## What to do next (Phase 2)
1. Capture real kiln photos from ~50-60 deg with the biochar surface + rim both visible.
2. Train/point the YOLO surface-segmenter at the kiln (the PoC ships a trained model + code).
3. Validate against a few weighed batches (load cell / known fills) to lock density + accuracy.
4. Optional hardware: fixed camera above the kiln, or a LiDAR phone (Polycam) / depth camera
   for a turnkey rig. The single-photo + known-geometry method is the core; hardware is polish.

## Honest caveats
- 2.7% is proven on synthetic data (only place true L/kg are known) + segmentation holds on
  real unseen containers (90% IoU). Real-charcoal field accuracy is the Phase-2 step.
- Single view can't see the back of a heap -> heaped surface is the main error term.
- The kiln model assumes a clean frustom fill; a small bottom cone (~100mm) is neglected
  (a few L near-empty); confirm density by weighing.
