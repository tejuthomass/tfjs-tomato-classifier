"""
Train + evaluate the two models and produce the report metrics.

    .venv/bin/python scripts/evaluate_models.py

  * 5-fold cross-validation on PlantVillage (lab), grouped by leaf ID (StratifiedGroupKFold),
    so photos of the same physical leaf never appear in both train and test.
  * Model A: Linear Discriminant Analysis on 4 colour features (baseline).
  * Model B: MobileNetV2 (alpha 0.35, ImageNet weights) fine-tuned.
  * Every fold's models are also tested on PlantDoc (field photos, never used for training)
    to measure the lab -> field accuracy drop.

Outputs:
  metrics/report_metrics.txt   <- numbers for the report
  metrics/fold_results.csv     <- raw per-fold numbers
  app/model/                   <- best CNN fold exported to TensorFlow.js + metadata.json
"""
import argparse
import json
import os
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KERAS_HOME", str(ROOT / ".tools" / "keras"))  # keep ImageNet weights inside project
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf

CLASSES = ["Healthy", "Early_Blight", "Late_Blight"]
IMG = 224

# Severity / colour thresholds. The web app reads these from app/model/metadata.json,
# so Python and JavaScript always use identical rules.
LEAF_MIN_SAT = 0.20      # pixels greyer than this are background
LEAF_MIN_VAL = 0.10      # pixels darker than this are shadow
GREEN_HUE = (65, 170)    # degrees; leaf pixels outside this range count as "non-green" (lesion/yellowing)
BANDS = [(10, "Low"), (30, "Medium"), (101, "High")]  # % non-green: <10 Low, 10-30 Medium, >=30 High


# --------------------------------------------------------------------------- images & features
def load_image(path):
    """Centre-crop to a square and resize to 224x224 (the app does the same with the camera frame)."""
    img = Image.open(ROOT / path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(img.resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)


def rgb_to_hsv(rgb):
    """rgb uint8 HxWx3 -> hue in degrees [0,360), saturation and value in [0,1]. Mirrors app.js."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    safe = np.where(d == 0, 1, d)
    h = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4))
    h = np.where(d == 0, 0, h * 60.0)
    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return h, s, mx


def color_features(rgb):
    """4 features for LDA: non-green fraction, mean hue, mean saturation, mean excess-green (leaf pixels only)."""
    h, s, v = rgb_to_hsv(rgb)
    leaf = (s > LEAF_MIN_SAT) & (v > LEAF_MIN_VAL)
    if leaf.sum() < 50:  # nothing leaf-like found: fall back to the whole image
        leaf = np.ones_like(leaf)
    green = (h >= GREEN_HUE[0]) & (h <= GREEN_HUE[1])
    x = rgb.astype(np.float32) / 255.0
    exg = 2 * x[..., 1] - x[..., 0] - x[..., 2]
    nongreen = 1.0 - (green & leaf).sum() / leaf.sum()
    return np.array([nongreen, h[leaf].mean() / 360.0, s[leaf].mean(), exg[leaf].mean()], dtype=np.float32)


def severity_band(nongreen_pct):
    for limit, name in BANDS:
        if nongreen_pct < limit:
            return name
    return BANDS[-1][1]


# --------------------------------------------------------------------------- CNN
def build_cnn():
    base = tf.keras.applications.MobileNetV2(input_shape=(IMG, IMG, 3), alpha=0.35, include_top=False,
                                             weights="imagenet", pooling="avg")
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="predictions")(x)
    return tf.keras.Model(base.input, out), base


def preprocess(x):
    """uint8 [0,255] -> float [-1,1] (MobileNetV2 convention). app.js does the same."""
    return tf.cast(x, tf.float32) / 127.5 - 1.0


def make_dataset(X, y, training, batch=32):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
        ds = ds.map(lambda a, b: (tf.image.random_flip_left_right(tf.image.random_flip_up_down(a)), b))
    ds = ds.map(lambda a, b: (preprocess(a), b))
    if training:
        ds = ds.map(lambda a, b: (tf.clip_by_value(tf.image.random_brightness(a, 0.15), -1, 1), b))
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


def train_cnn(X, y, epochs_head, epochs_fine):
    model, base = build_cnn()
    # Stage 1: train only the new classifier head
    base.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
    model.fit(make_dataset(X, y, True), epochs=epochs_head, verbose=0)
    # Stage 2: fine-tune the last 30 layers (BatchNorm stays frozen - standard practice for small data)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(tf.keras.optimizers.Adam(1e-4), "sparse_categorical_crossentropy", ["accuracy"])
    model.fit(make_dataset(X, y, True), epochs=epochs_fine, verbose=0)
    return model


def cnn_predict(model, X):
    return model.predict(make_dataset(X, np.zeros(len(X)), False), verbose=0).argmax(1)


# --------------------------------------------------------------------------- helpers
def scores(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)))


def median_latency_ms(fn, sample, runs=100, warmup=10):
    for _ in range(warmup):
        fn(sample)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn(sample)
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def fmt_cm(cm):
    w = max(len(c) for c in CLASSES) + 2
    lines = ["true \\ pred".ljust(w) + "".join(c.rjust(w) for c in CLASSES)]
    for c, row in zip(CLASSES, cm):
        lines.append(c.ljust(w) + "".join(str(v).rjust(w) for v in row))
    return "\n".join("    " + l for l in lines)


def fmt_recall(y_true, y_pred):
    r = recall_score(y_true, y_pred, average=None, labels=range(len(CLASSES)), zero_division=0)
    return "\n".join(f"    {c:<14} {v:.3f}" for c, v in zip(CLASSES, r))


def ms(values):
    return f"{np.mean(values):.3f} ± {np.std(values, ddof=1):.3f}"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs-head", type=int, default=6)
    ap.add_argument("--epochs-fine", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    tf.keras.utils.set_random_seed(args.seed)

    manifest = pd.read_csv(ROOT / "data" / "manifest.csv")
    lab = manifest[manifest.dataset == "plantvillage"].reset_index(drop=True)
    field = manifest[manifest.dataset == "plantdoc"].reset_index(drop=True)

    print(f"Loading {len(lab)} lab + {len(field)} field images ...")
    X_lab = np.stack([load_image(p) for p in lab.path])
    X_field = np.stack([load_image(p) for p in field.path])
    y_lab = lab.label.map(CLASSES.index).to_numpy()
    y_field = field.label.map(CLASSES.index).to_numpy()
    groups = lab.leaf_id.to_numpy()
    F_lab = np.stack([color_features(x) for x in X_lab])
    F_field = np.stack([color_features(x) for x in X_field])

    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows, overlap = [], 0
    oof = {"LDA": np.zeros(len(lab), int), "CNN": np.zeros(len(lab), int)}
    best = {"LDA": (-1, None, None), "CNN": (-1, None, None)}  # (lab F1, model, field predictions)

    for k, (tr, te) in enumerate(cv.split(X_lab, y_lab, groups), 1):
        overlap += len(set(groups[tr]) & set(groups[te]))

        lda = LinearDiscriminantAnalysis().fit(F_lab[tr], y_lab[tr])
        cnn = train_cnn(X_lab[tr], y_lab[tr], args.epochs_head, args.epochs_fine)

        for name, pred_lab, pred_field, model in [
            ("LDA", lda.predict(F_lab[te]), lda.predict(F_field), lda),
            ("CNN", cnn_predict(cnn, X_lab[te]), cnn_predict(cnn, X_field), cnn),
        ]:
            oof[name][te] = pred_lab
            acc, f1 = scores(y_lab[te], pred_lab)
            facc, ff1 = scores(y_field, pred_field)
            rows.append(dict(fold=k, model=name, lab_acc=acc, lab_f1=f1, field_acc=facc, field_f1=ff1,
                             n_train=len(tr), n_test=len(te)))
            if f1 > best[name][0]:
                best[name] = (f1, model, pred_field)
            print(f"fold {k} {name}: lab acc {acc:.3f} F1 {f1:.3f} | field acc {facc:.3f} F1 {ff1:.3f}")

    res = pd.DataFrame(rows)
    (ROOT / "metrics").mkdir(exist_ok=True)
    res.to_csv(ROOT / "metrics" / "fold_results.csv", index=False)

    # Latency: one image at a time (batch size 1), CPU, including preprocessing / feature extraction
    lda_best, cnn_best = best["LDA"][1], best["CNN"][1]
    sample = X_lab[0]
    lat_lda = median_latency_ms(lambda x: lda_best.predict(color_features(x)[None]), sample)
    cnn_fn = tf.function(lambda t: cnn_best(t, training=False))
    lat_cnn = median_latency_ms(lambda x: cnn_fn(preprocess(x[None])).numpy(), sample)

    # Severity sanity check on lab images: band distribution per true class
    sev = pd.DataFrame({"label": lab.label, "band": [severity_band(100 * f[0]) for f in F_lab]})
    sev_table = pd.crosstab(sev.label, sev.band).reindex(index=CLASSES, columns=[b for _, b in BANDS], fill_value=0)

    # ------------------------------------------------------------------- write report
    test_only = (field.split == "test").to_numpy()
    out = []
    w = out.append
    w("TOMATO LEAF DISEASE SCREENING - REPORT METRICS")
    w(f"Generated: {datetime.now():%Y-%m-%d %H:%M}   seed={args.seed}")
    w("=" * 72)
    w("\nDATA")
    w(f"  Lab (PlantVillage): {len(lab)} images, {len(set(groups))} unique leaf IDs  "
      + ", ".join(f"{c}={n}" for c, n in lab.label.value_counts().reindex(CLASSES).items()))
    w(f"  Field (PlantDoc, never used for training): {len(field)} images  "
      + ", ".join(f"{c}={n}" for c, n in field.label.value_counts().reindex(CLASSES).items()))
    w(f"  Cross-validation: {args.folds}-fold StratifiedGroupKFold grouped by leaf ID")
    w(f"  Leaf IDs shared between train and test across all folds: {overlap}  (must be 0)")
    w("\nMODELS")
    w("  LDA baseline : Linear Discriminant Analysis on 4 colour features")
    w("                 (non-green fraction, mean hue, mean saturation, mean excess-green)")
    w(f"  CNN          : MobileNetV2 alpha=0.35, ImageNet init, {args.epochs_head} head + "
      f"{args.epochs_fine} fine-tune epochs, {IMG}x{IMG} input")

    for name in ["LDA", "CNN"]:
        r = res[res.model == name]
        w("\n" + "=" * 72)
        w(f"{name}")
        w("=" * 72)
        w(f"  Lab   (PlantVillage, {args.folds}-fold CV)  Accuracy  {ms(r.lab_acc)}")
        w(f"                                       Macro-F1  {ms(r.lab_f1)}")
        w(f"  Field (PlantDoc, each fold's model)  Accuracy  {ms(r.field_acc)}")
        w(f"                                       Macro-F1  {ms(r.field_f1)}")
        w(f"  Lab -> field accuracy drop: {100 * (r.lab_acc.mean() - r.field_acc.mean()):.1f} percentage points")
        w(f"\n  Per-class recall, lab (pooled out-of-fold predictions):")
        w(fmt_recall(y_lab, oof[name]))
        w(f"  Confusion matrix, lab (pooled out-of-fold, rows = true class):")
        w(fmt_cm(confusion_matrix(y_lab, oof[name], labels=range(len(CLASSES)))))
        pf = best[name][2]
        a_all, _ = scores(y_field, pf)
        a_test, _ = scores(y_field[test_only], pf[test_only])
        w(f"\n  Best fold model (highest lab F1) on field images:")
        w(f"    accuracy all PlantDoc = {a_all:.3f} (n={len(field)}); "
          f"official PlantDoc test split only = {a_test:.3f} (n={test_only.sum()})")
        w(f"  Per-class recall, field:")
        w(fmt_recall(y_field, pf))
        w(f"  Confusion matrix, field (rows = true class):")
        w(fmt_cm(confusion_matrix(y_field, pf, labels=range(len(CLASSES)))))

    w("\n" + "=" * 72)
    w("MEDIAN INFERENCE LATENCY (single image, batch size 1, CPU, 100 runs after 10 warm-up)")
    w(f"  Machine: {platform.platform()} / {platform.processor() or platform.machine()}")
    w(f"  LDA (feature extraction + predict): {lat_lda:.2f} ms")
    w(f"  CNN (preprocess + forward pass)   : {lat_cnn:.2f} ms")
    w("  (On-phone TF.js latency is shown inside the app after each scan.)")

    w("\n" + "=" * 72)
    w("SEVERITY BANDS on lab images (% non-green leaf pixels; <10 Low, 10-30 Medium, >=30 High)")
    w("  Sanity check only - PlantVillage has no severity labels.")
    w("    " + sev_table.to_string().replace("\n", "\n    "))
    w("\nNotes")
    w("  * Mean ± SD uses the sample SD (ddof=1) across folds.")
    w("  * PlantVillage has no official leaf ID; it is derived from the source photo code in each")
    w("    file name (repeat shots '.1/.2' and augmentation suffixes are merged).")
    w("  * Small sample sizes -> treat numbers as indicative, not definitive.")
    report = "\n".join(out) + "\n"
    (ROOT / "metrics" / "report_metrics.txt").write_text(report)
    print("\n" + report)

    # ------------------------------------------------------------------- export best CNN to TF.js
    import tensorflowjs as tfjs  # imported late: slow import and noisy warnings
    model_dir = ROOT / "app" / "model"
    shutil.rmtree(model_dir, ignore_errors=True)
    tfjs.converters.save_keras_model(cnn_best, str(model_dir))
    r = res[res.model == "CNN"]
    meta = dict(labels=CLASSES, input_size=IMG, preprocessing="x / 127.5 - 1 (centre-crop square, RGB)",
                severity=dict(leaf_min_sat=LEAF_MIN_SAT, leaf_min_val=LEAF_MIN_VAL, green_hue=list(GREEN_HUE),
                              bands=[dict(below=l, name=n) for l, n in BANDS]),
                metrics=dict(lab_accuracy=round(r.lab_acc.mean(), 3), field_accuracy=round(r.field_acc.mean(), 3)),
                exported=f"{datetime.now():%Y-%m-%d}")
    (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Exported best CNN fold (lab F1 {best['CNN'][0]:.3f}) to {model_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
