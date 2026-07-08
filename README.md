# Kon-Tiki Biochar Volume Estimation

Estimate how much biochar (in **litres**) is inside a **Kon-Tiki 1000** kiln from a single
phone photo taken after quenching.

This folder contains the working system, the trained model, the training code, the
geometry-based method, and an honest account of the accuracy.

---

## 1. What the system does

You upload **one photo** of the biochar surface inside the kiln. The system returns:

- estimated **volume in litres** (and approximate weight in kg),
- a **confidence range**,
- an **attention heatmap** showing which pixels the model used,
- the **isolated kiln** (background removed),
- **capture-quality checks** that warn if the photo is blurry / dark / bad angle.

The web app is [app.py](app.py).

---

## 2. How the number is produced (two methods)

### Method A — Learned model (what the web app runs)

```
photo
  -> SAM (mobile_sam.pt) masks the image to the KILN ONLY  (removes background, people, ground)
  -> EfficientNet-B0 CNN (kontiki_cnn_masked.pt) reads the biochar fill and predicts litres
  -> 6 augmented views give a prediction spread  = confidence range
  -> Grad-CAM heatmap shows WHERE the model looked (should be on the biochar)
```

Trained on ~8,000 real, manually-measured kiln photos.
Training code: [gpu_train_masked.py](gpu_train_masked.py) (runs on a free Google Colab GPU).

### Method B — Geometry (physics-based; the most reliable method)

```
photo -> segment the biochar SURFACE circle and the kiln RIM circle
      -> u = surface radius / rim radius   (a scale-free ratio -- the kiln is the ruler)
      -> known Kon-Tiki 1000 cone shape -> fill height -> litres
```

Calculator + geometry: [kiln_volume_calc.py](kiln_volume_calc.py).
The full **proof of concept** for this method (results, charts, notebook, and code showing
*why* it was chosen) is in the [charcoal_poc_share/](charcoal_poc_share/) folder.
Kon-Tiki 1000 dimensions (from the design drawing): rim Ø1500 mm, bottom Ø803 mm, depth
930 mm → computed full capacity **998 L** vs the 1000 L spec = **99.8% match** (validates the dims).

---

## 3. Accuracy (honest, measured)

| Method | Error | Notes |
|---|---|---|
| Learned model — held-out test (n = 2,339 unseen photos) | **~3.3% average**, 98% within 10% | Tracks the fill level (corr 0.71). |
| Geometry method — proof of concept | **~2.7%** | Segmentation ~90% IoU on unseen containers. |

**Key limitation (be transparent about this):** a single 2-D photo cannot resolve very
small differences in fill. The manual ground-truth volumes were rounded to the nearest
~10 L (±25 L), so the model cannot reliably tell apart two kilns differing by less than
~28 L, and accuracy depends heavily on how the photo is taken (angle, lighting, framing).
For higher and more consistent accuracy, the recommended next step is a **multi-view (slow
video) or depth-camera** capture — see [FINAL_RECOMMENDATION.md](FINAL_RECOMMENDATION.md).

---

## 4. Files

| File | What it is |
|---|---|
| [app.py](app.py) | The web app: upload a photo → volume + full breakdown. Runs locally on CPU. |
| `kontiki_cnn_masked.pt` | The trained CNN model the app uses. |
| `mobile_sam.pt` | Segment-Anything model used to isolate the kiln from the background. |
| [gpu_train_masked.py](gpu_train_masked.py) | Colab training script that produced the model. |
| [kiln_volume_calc.py](kiln_volume_calc.py) | Geometry method + Kon-Tiki 1000 volume table. |
| `sample_kiln_images/` | A few labelled photos to test the app with. |
| `charcoal_poc_share/` | Proof of concept — results, notebook, charts and code showing why this method was chosen. |
| [FINAL_RECOMMENDATION.md](FINAL_RECOMMENDATION.md) | Recommended method + path to higher accuracy. |
| [POC_REPORT.md](POC_REPORT.md) | Proof-of-concept report. |
| [start_demo.bat](start_demo.bat) | One-click launcher for the web app (Windows). |
| [requirements.txt](requirements.txt) | Python dependencies. |

---

## 5. How to run it

Requires **Python 3.10**.

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in a browser and upload a photo
(try one from `sample_kiln_images/`).

On Windows you can simply double-click **start_demo.bat**.

> Note: the trained model (`kontiki_cnn_masked.pt`) is included in the repo. The kiln-masking
> model `mobile_sam.pt` downloads automatically on first run (needs internet once), then the
> models load in ~10–20 seconds.
