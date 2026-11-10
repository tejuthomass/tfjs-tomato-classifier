"""
One-command setup for the Tomato Leaf Screening project.

    python3 scripts/setup.py              # default: 150 PlantVillage images per class
    python3 scripts/setup.py --pv-per-class 300

What it does (everything stays inside this project folder):
  1. Makes sure ./.venv exists and runs Python 3.11 (TensorFlow does not support 3.14 yet).
     If your system Python is not 3.10/3.11, a private Python 3.11 is downloaded into ./.tools/
  2. Installs the pinned pip packages into ./.venv
  3. Downloads a small sample of PlantVillage (lab) and PlantDoc (field) images
     for Tomato Healthy / Early Blight / Late Blight into ./data/
  4. Downloads TensorFlow.js (tf.min.js) into ./app/vendor/ so the app works offline
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
TOOLS = ROOT / ".tools"
VENV_PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# Packages installed normally
PACKAGES = [
    "tensorflow==2.15.1",
    "numpy==1.26.4",
    "scikit-learn==1.9.1",
    "pandas==3.0.6",
    "pillow==13.0.0",
    "setuptools<70",          # tensorflow_hub still imports pkg_resources
    "tensorflow-hub==0.16.1",
    "importlib_resources==7.1.0",
]
# The TF.js converter declares heavy optional dependencies (flax, newer jax, ...) that
# conflict with TF 2.15. We install only what its import chain really needs, without deps.
NO_DEPS_PACKAGES = [
    "tensorflowjs==4.22.0",
    "tensorflow-decision-forests==1.8.1",
    "jax==0.4.23",
    "jaxlib==0.4.23",
]

TFJS_URL = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"

CLASSES = ["Healthy", "Early_Blight", "Late_Blight"]
PLANTVILLAGE = {  # repo: spMohanty/PlantVillage-Dataset, folder raw/color/
    "Healthy": "Tomato___healthy",
    "Early_Blight": "Tomato___Early_blight",
    "Late_Blight": "Tomato___Late_blight",
}
PLANTDOC = {  # repo: pratikkayal/PlantDoc-Dataset, folders train/ and test/
    "Healthy": "Tomato leaf",
    "Early_Blight": "Tomato Early blight leaf",
    "Late_Blight": "Tomato leaf late blight",
}


# --------------------------------------------------------------------------- 1. venv
def ensure_venv():
    """Re-run this script inside ./.venv (Python 3.11), creating it first if needed."""
    in_venv = Path(sys.prefix).resolve() == VENV.resolve()
    if in_venv and sys.version_info[:2] in [(3, 10), (3, 11)]:
        return

    ok = False
    if VENV_PY.exists():
        ver = subprocess.run([str(VENV_PY), "-c", "import sys;print(sys.version_info[:2])"],
                             capture_output=True, text=True).stdout.strip()
        ok = ver in ("(3, 10)", "(3, 11)")
        if not ok:
            backup = TOOLS / "old_venv"
            print(f"[setup] .venv uses Python {ver}; TensorFlow needs 3.10/3.11. Moving it to {backup}")
            TOOLS.mkdir(exist_ok=True)
            if backup.exists():
                import shutil
                shutil.rmtree(backup)
            VENV.rename(backup)

    if not ok:
        if sys.version_info[:2] in [(3, 10), (3, 11)]:
            print("[setup] Creating .venv with", sys.executable)
            subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])
        else:
            # Use 'uv' (installed into a private helper venv) to fetch a local Python 3.11.
            print("[setup] Downloading a private Python 3.11 into .tools/ (system Python is",
                  ".".join(map(str, sys.version_info[:2])) + ")")
            helper = TOOLS / "bootstrap"
            helper_bin = helper / ("Scripts" if os.name == "nt" else "bin")
            if not helper.exists():
                subprocess.check_call([sys.executable, "-m", "venv", str(helper)])
            subprocess.check_call([str(helper_bin / "python"), "-m", "pip", "install", "-q", "uv"])
            env = dict(os.environ, UV_PYTHON_INSTALL_DIR=str(TOOLS / "python"))
            uv = str(helper_bin / "uv")
            subprocess.check_call([uv, "python", "install", "3.11"], env=env)
            subprocess.check_call([uv, "venv", "--seed", "--python", "3.11", str(VENV)], env=env)

    print("[setup] Re-running inside .venv ...")
    sys.exit(subprocess.call([str(VENV_PY), __file__, *sys.argv[1:]]))


# --------------------------------------------------------------------------- 2. pip
def install_packages():
    pip = [sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check"]
    print("[setup] Installing packages (first run downloads ~600 MB) ...")
    subprocess.check_call(pip + PACKAGES)
    subprocess.check_call(pip + ["--no-deps"] + NO_DEPS_PACKAGES)


# --------------------------------------------------------------------------- 3. data
def github_list(repo, folder):
    url = f"https://api.github.com/repos/{repo}/contents/{urllib.parse.quote(folder)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        items = json.load(r)
    return [i for i in items if i["type"] == "file"]


def leaf_id_from_plantvillage(filename, label):
    """
    PlantVillage names look like  '<uuid>___GH_HL Leaf 259.1.JPG'  or  '<uuid>___RS_Erly.B 8389.JPG'.
    The part after '___' is the source photo code; '.1', '.2' ... are repeat shots of the same
    leaf and suffixes like '_flipLR' are augmented copies. We strip those to get a leaf ID.
    (PlantVillage does not publish an official leaf ID, so this is the best available proxy.)
    """
    code = filename.split("___", 1)[-1].rsplit(".", 1)[0]
    code = re.sub(r"(_(flip\w*|\d+deg\w*|new\w*|change\w*))+$", "", code, flags=re.I)
    code = re.sub(r"\.\d+$", "", code.strip())
    return f"{label}:{code}"


def fetch(url):
    """Try the jsDelivr GitHub mirror first (much faster than raw.githubusercontent.com), then GitHub."""
    m = re.match(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url)
    urls = [f"https://cdn.jsdelivr.net/gh/{m[1]}/{m[2]}@{m[3]}/{m[4]}", url] if m else [url]
    for i, u in enumerate(urls):
        try:
            with urllib.request.urlopen(u, timeout=60) as r:
                return r.read()
        except Exception:
            if i == len(urls) - 1:
                raise


def download(url, dest, max_side=None):
    if dest.exists() or (max_side and dest.with_suffix(".jpg").exists()):
        return True
    try:
        data = fetch(url)
        dest.write_bytes(data)
        if max_side:  # shrink big field photos to save disk space
            from PIL import Image
            img = Image.open(dest).convert("RGB")
            img.thumbnail((max_side, max_side))
            img.save(dest.with_suffix(".jpg"), quality=90)
            if dest.suffix.lower() != ".jpg":
                dest.unlink()
        return True
    except Exception as e:  # a few PlantDoc files are broken links / not images
        print(f"   ! skipped {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_data(pv_per_class, seed=42):
    rng = random.Random(seed)
    rows, jobs = [], []
    data = ROOT / "data"

    for label, folder in PLANTVILLAGE.items():
        files = github_list("spMohanty/PlantVillage-Dataset", f"raw/color/{folder}")
        files = rng.sample(files, min(pv_per_class, len(files)))
        out = data / "plantvillage" / label
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            dest = out / f["name"]
            jobs.append((f["download_url"], dest, None))
            rows.append(dict(path=str(dest.relative_to(ROOT)), dataset="plantvillage", split="lab",
                             label=label, leaf_id=leaf_id_from_plantvillage(f["name"], label)))
        print(f"[data] PlantVillage {label}: {len(files)} images")

    for split in ["train", "test"]:  # none of PlantDoc is used for training: it is the field test set
        for label, folder in PLANTDOC.items():
            files = [f for f in github_list("pratikkayal/PlantDoc-Dataset", f"{split}/{folder}")
                     if f["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
            out = data / "plantdoc" / split / label
            out.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(files):
                dest = out / f"{i:04d}{Path(f['name']).suffix.lower()}"
                jobs.append((f["download_url"], dest, 512))
                rows.append(dict(path=str(dest.with_suffix(".jpg").relative_to(ROOT)), dataset="plantdoc",
                                 split=split, label=label, leaf_id=f"plantdoc:{split}:{label}:{i}"))
            print(f"[data] PlantDoc {split}/{label}: {len(files)} images")

    print(f"[data] Downloading {len(jobs)} images ...")
    with ThreadPoolExecutor(16) as pool:
        ok = list(pool.map(lambda j: download(*j), jobs))

    rows = [r for r, good in zip(rows, ok) if good]
    with open(data / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[data] Wrote data/manifest.csv ({len(rows)} images)")


def download_tfjs():
    dest = ROOT / "app" / "vendor" / "tf.min.js"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("[setup] Downloading TensorFlow.js 4.22.0 for offline use ...")
        urllib.request.urlretrieve(TFJS_URL, dest)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv-per-class", type=int, default=150, help="PlantVillage images per class")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    ensure_venv()
    if not args.skip_install:
        install_packages()
    download_data(args.pv_per_class)
    download_tfjs()
    print("\n[setup] Done. Next:  .venv/bin/python scripts/evaluate_models.py")
