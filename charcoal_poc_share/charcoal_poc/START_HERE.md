# Charcoal Volume/Weight POC — how to run & show this

You have **three ways** to see the work, easiest first. You do **not** need a GPU and
you do **not** need to retrain anything — the results and trained models are included.

---

## Option 1 — Just SHOW the results (easiest, 1 minute, recommended)

Open **`POC_results.ipynb`** in Google Colab:
1. Go to https://colab.research.google.com → **File → Upload notebook** → pick `POC_results.ipynb`
2. **Runtime → Run all**

Everything (charts, example images, scorecard, camera-angle study) appears in ~10
seconds. Nothing to install — it's fully self-contained. **This is the one to present.**

*(It also opens in plain Jupyter on a laptop: `pip install notebook && jupyter notebook`.)*

---

## Option 2 — Run the pipeline LIVE on a normal laptop (CPU, no GPU)

Proves the system actually computes — not just baked images. From this folder:

```bash
pip install numpy opencv-python-headless          # tiny, no GPU
python demo_cpu.py
```

It segments 60 held-out test images, estimates volume & weight, and compares to the
true answers — printing ~2.7% error for the AI ceiling vs ~16% for the no-AI baseline.

Optional (runs the trained AI too; bigger install):
```bash
pip install ultralytics
python demo_cpu.py --yolo
```

> Have **Claude Code**? Just open this folder in it and say *"run the CPU demo and
> explain the results"* — it knows these files.

---

## Option 3 — Re-create everything from scratch (needs a GPU, ~40 min)

Only if someone wants full reproduction (render new 3-D data + retrain). This needs a
GPU machine (e.g. Colab **GPU** runtime) and the Blender engine. See `README.md` →
"Reproduce". Not needed just to show the work.

---

## What's in this folder

| Path | What |
|---|---|
| `POC_results.ipynb` | self-contained results notebook (Option 1) |
| `demo_cpu.py` | live CPU demo (Option 2) |
| `docs/POC_REPORT.md` | the full written report (incl. camera-angle study) |
| `results/figures/` | all charts & the explainer/angle images |
| `results/*.json` | raw metrics |
| `weights/` | the two trained models (synthetic fill + real container) |
| `demo_data/` | 60 held-out test images + ground truth (for the CPU demo) |
| `src/` | the pipeline code |

## The headline (for the team)
- On images with known truth: **~2.7%** volume **and** weight error.
- AI traced the contents correctly: **~98%** (synthetic) / **~90%** (real, unseen containers).
- All 5 POC accuracy targets met. A learned AI is essential; the heaped pile is the main
  error term; best camera angle is **~50–60° looking down** into the drum.
- Absolute accuracy is proven on synthetic data (only place the true litres/kg are known);
  real-charcoal field accuracy is the Phase-2 step (real images + load cell).
