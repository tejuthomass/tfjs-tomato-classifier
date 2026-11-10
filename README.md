# Tomato Leaf Check: offline screening and report metrics

**Live Demo:** https://rococo-naiad-cf4ef1.netlify.app/

This project has two parts:

1. **A phone app** that sorts a tomato leaf photo into **Healthy, Early Blight or Late Blight**. It estimates severity (Low / Medium / High) and shows advice in **English and Hindi**. It runs fully offline in the phone's browser using TensorFlow.js.
2. **An evaluation script** that writes every statistic the Research Methodology report needs to `metrics/report_metrics.txt`.

```
tomato_project/
├── scripts/
│   ├── setup.py              # creates .venv, installs packages, downloads data
│   └── evaluate_models.py    # cross-validation, LDA vs CNN, field test, latency, TF.js export
├── data/                     # downloaded images + manifest.csv (created by setup.py)
├── metrics/                  # report_metrics.txt + fold_results.csv (created by evaluate_models.py)
├── report_figures/           # app screenshots for the report (see report_figures/README.md)
└── app/                      # the offline web app (PWA)
    ├── index.html            # page + all CSS
    ├── app.js                # camera, TF.js inference, severity, advisory card
    ├── sw.js                 # service worker (offline cache)
    ├── manifest.json         # makes it installable
    ├── icons/  vendor/tf.min.js  model/   (model/ is created by evaluate_models.py)
```

---

## Step 1: Setup (one time, about 10–15 min)

Open Terminal in this folder and run:

```bash
python3 scripts/setup.py
```

This installs everything inside the project folder and nothing system-wide:

* **`.venv` with Python 3.11.** TensorFlow does not support Python 3.14 yet. If your computer only has a newer Python, a private Python 3.11 is downloaded into `.tools/`.
* **Packages.** TensorFlow 2.15, scikit-learn, pandas, Pillow, and the TensorFlow.js converter.
* **Datasets.** 150 random PlantVillage images per class go to `data/plantvillage/`. All PlantDoc tomato images for the 3 classes go to `data/plantdoc/`. The list of images is written to `data/manifest.csv`.
* **The TF.js library.** It is saved to `app/vendor/tf.min.js` so the app never needs the internet.

Want more lab images? Run `python3 scripts/setup.py --pv-per-class 300`.

## Step 2: Generate the report metrics (a few minutes on a laptop)

```bash
.venv/bin/python scripts/evaluate_models.py
```

What it does:

| Item | Method |
|---|---|
| Cross-validation | 5-fold `StratifiedGroupKFold` on PlantVillage, **grouped by leaf ID**, so the same leaf is never in both train and test. The report prints the overlap count, which should be 0. |
| Baseline | **LDA** on 4 colour features: non-green fraction, mean hue, mean saturation, mean excess-green |
| CNN | **MobileNetV2** (α = 0.35, ImageNet weights). The new head is trained first, then the last 30 layers are fine-tuned. |
| Field test | Every fold's models are also tested on **PlantDoc** field photos, which are never used for training. This gives the lab → field accuracy drop. |
| Latency | Median time to classify one image (batch size 1, CPU, 100 runs) |

Outputs:

* `metrics/report_metrics.txt` has:
  * Mean ± SD accuracy and macro-F1 across the 5 folds, for lab and field.
  * Per-class precision, recall and F1, plus confusion matrices. The lab tables pool all 5 folds' out-of-fold predictions. The **field tables come from a single fold model**, the fold with the highest lab F1.
  * Median latency and severity-band counts.
* `metrics/fold_results.csv` has the raw per-fold numbers, if you want to make charts.
* `app/model/` is the best CNN fold (highest lab F1) exported to TensorFlow.js, plus `metadata.json`.

> You may see the warning `TensorFlow Decision Forests 1.8.1 is compatible with ... 2.15.0`. It is harmless. That package is only needed so the TF.js converter can be imported.

## Step 3: Run the app on your computer

```bash
cd app
../.venv/bin/python -m http.server 8000
```

Open **http://localhost:8000** in Chrome. Click **Start camera** (or **Upload photo**), then **Scan leaf**.
Stop the server with `Ctrl+C`.

## Step 4: Put it on your phone and use it offline

Browsers only allow the **camera** and the **offline cache (service worker)** on `https://` pages or on `localhost`.

**Easiest:** open the live demo on your phone at **https://rococo-naiad-cf4ef1.netlify.app/**, then follow [Install to the home screen](#install-to-the-home-screen).

To run your own copy instead, pick one of these options.

### Option A: Android + USB cable (no internet hosting needed)
1. On the phone, enable **Developer options → USB debugging**, then plug it into the laptop.
2. Keep the server from Step 3 running.
3. On the laptop, open Chrome at `chrome://inspect/#devices` → **Port forwarding…** → add `8000` → `localhost:8000` → tick **Enable port forwarding**.
4. On the phone, open Chrome at **http://localhost:8000**.

### Option B: Free HTTPS hosting (works on iPhone and Android)
The live demo is hosted this way on Netlify: **https://rococo-naiad-cf4ef1.netlify.app/**

* **Update the live demo:** in the Netlify dashboard, open the site → **Deploys**, then drag the whole `app` folder onto the drop area. Bump the cache version first (see [Updating the app after changes](#updating-the-app-after-changes)).
* **New site of your own:** drag the `app` folder onto **https://app.netlify.com/drop**, or push it to a GitHub Pages repo. Open the `https://…` link it gives you on your phone.

(The hosted app is public. It contains no personal data, but be aware of it.)

### Option C: Same Wi-Fi, quick test only
```bash
cd app && ../.venv/bin/python -m http.server 8000 --bind 0.0.0.0
ipconfig getifaddr en0        # prints your laptop's IP, e.g. 192.168.1.20
```
Open `http://192.168.1.20:8000` on the phone. **Upload photo** works because it can open the camera app. Live camera and offline mode will **not** work over plain http.

### Install to the home screen
* **Android (Chrome):** menu ⋮ → **Install app** or **Add to Home screen**.
* **iPhone (Safari):** Share button → **Add to Home Screen**.

### Airplane Mode demo checklist
1. Open the app once while online and wait until the top-right pill says **Model ready**. This means everything is cached.
2. Turn on **Airplane Mode**.
3. Close the app completely, then open it from the home screen icon.
4. Scan a leaf. Everything (TF.js, model, advice) loads from the phone's cache.

### Updating the app after changes
The service worker serves files from its cache first. After you edit any file in `app/` or re-run `evaluate_models.py`, increase the number in `CACHE = 'leaf-check-v3'` in `app/sw.js` (v3 → v4 → v5 …). Reload the page twice.

---

## How severity is calculated

1. **Find the leaf.** Pixels with HSV saturation > 0.20 and brightness > 0.10 count as leaf. Grey or white background and deep shadow are ignored.
2. **Measure the non-green area.** Leaf pixels whose hue falls outside 65°–170° count as **non-green** (lesions or yellowing). Severity is the non-green percentage of the leaf.
3. **Assign a band.** **Low** < 10 %, **Medium** 10–30 %, **High** ≥ 30 %.

Python and the app read the same thresholds from `app/model/metadata.json`.
Tick **Highlight non-green areas** in the app to see which pixels were counted.
This is a coarse colour heuristic, not a calibrated measurement. Severity is not shown for leaves classified as Healthy.
**Known limitation:** grey or white dead tissue at the centre of a lesion has low saturation, so it is counted as background. That means severity tends to be *under*-estimated for advanced lesions, and field photos with busy backgrounds are noisy.

## Notes for the methodology / limitations section

* **Leaf IDs.** PlantVillage does not publish physical leaf IDs. The ID is taken from the source photo code in each file name (e.g. `GH_HL Leaf 259.1` → `GH_HL Leaf 259`). Repeat shots (`.1`, `.2`) and augmented copies (`_flipLR`, …) are merged into one group.
* **Field set.** All PlantDoc tomato images for these 3 classes, from both its `train` and `test` folders, form the field set. None are used for training. The field set has 245 images: 61 Healthy, 81 Early Blight, 103 Late Blight. The report also gives accuracy on PlantDoc's official `test` split alone (n = 26: 8 Healthy, 9 Early Blight, 9 Late Blight). The repository lists 27 test images, but one link returned 404 during download. `data/manifest.csv` lists exactly the images that were evaluated.
* **Sample size.** The sample is small (450 lab images), so present the numbers as indicative and quote the SDs.
* **Model selection.** The exported app model is the best of the 5 fold models. The CV mean ± SD is the unbiased estimate to report, not the best fold.
* **Single-fold vs 5-fold field results.** The field confusion matrices and per-class field metrics come from **one** fold model: CNN fold 4 and LDA fold 5. On all 245 field images, those single models score **50.2 %** (CNN) and **34.3 %** (LDA). The **5-fold mean** field accuracies are **49.1 % ± 1.2** (CNN) and **35.3 % ± 0.9** (LDA). Quote the 5-fold means as the headline result and label the matrices as single-fold.
* **Per-class precision / F1.** These are calculated directly from the confusion matrices. The CNN field model never predicts Healthy, so its Healthy precision is undefined (0/0) and is reported as n/a, not 0.
* **Fixed epochs.** There was no hyper-parameter tuning on test folds. Epochs are fixed (6 head + 4 fine-tune).
* **Random seed.** Seed 42 is used throughout. Re-running gives very similar, though not bit-identical, CNN numbers.

## Sources

* Data: [PlantVillage dataset](https://github.com/spMohanty/PlantVillage-Dataset) (Hughes & Salathé, 2015) and [PlantDoc dataset](https://github.com/pratikkayal/PlantDoc-Dataset) (Singh et al., 2020).
* Early blight advice: [NC State Extension: Early Blight of Tomato](https://content.ces.ncsu.edu/early-blight-of-tomato) (Meadows, Hinchliffe & Villani, revised 2025).
* Late blight advice: [Washington State University HortSense: Tomato, Late blight](https://hortsense.cahnrs.wsu.edu/fact-sheet/tomato-late-blight/).
* TF.js export: [Importing a Keras model into TensorFlow.js](https://www.tensorflow.org/js/tutorials/conversion/import_keras).
* Offline caching: [MDN: Using Service Workers](https://developer.mozilla.org/en-US/docs/Web/API/Service_Worker_API/Using_Service_Workers).

The Hindi advisory text is a translation of the English. Have a native speaker or extension officer review it before field use. To use a different language, edit the `hi` block in `app/app.js` and the button label in `index.html`.
