#!/usr/bin/env python3
"""Run any of the Kaggle notebooks in experiments/notebooks/ on your own machine.

The notebooks are kept exactly as they were executed for the paper. This runner
extracts the experiment cell, points it at your local copy of the ALASKA#2 cover
images and at an output folder, and executes it.

    python experiments/run_local.py --notebook experiments/notebooks/k8a_one_factor_ablation.ipynb \
        --images /data/alaska2/Cover --out runs/k8a
    # add --quick for a small-scale smoke run (proof mode: fewer images, 128-px crops)
"""
import argparse, json, os, re, runpy, sys, tempfile

# Python salts str hashes per process; several notebooks derive per-image seeds
# from hash(). Re-launch with a fixed PYTHONHASHSEED so a local run is repeatable.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

ap = argparse.ArgumentParser()
ap.add_argument("--notebook", required=True)
ap.add_argument("--images", required=True, help="folder with the ALASKA#2 cover JPEGs")
ap.add_argument("--out", required=True, help="output folder for CSVs, tables, figures")
ap.add_argument("--quick", action="store_true", help="proof-mode smoke run")
a = ap.parse_args()

nb = json.load(open(a.notebook, encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
code = max(cells, key=len)                       # the experiment cell
os.makedirs(a.out, exist_ok=True)
code = re.sub(r'^IMAGE_DIR\s*=.*$', f'IMAGE_DIR = {os.path.abspath(a.images)!r}', code, flags=re.M)
code = re.sub(r'^OUT_DIR\s*=.*$',   f'OUT_DIR   = {os.path.abspath(a.out)!r}',    code, flags=re.M)
if a.quick:
    code = re.sub(r'^(MODE|REV_MODE|K6_MODE)(\s*)=\s*"full"', r'\1\2= "proof"', code, flags=re.M)
    code = re.sub(r'^assert .*full.*$', "pass", code, flags=re.M)
    code = re.sub(r'^assert RV_N_STEG.*$', "pass", code, flags=re.M)
    code = re.sub(r'^assert RV_CROP.*$', "pass", code, flags=re.M)
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
    f.write(code); path = f.name
print(f"running {os.path.basename(a.notebook)}  images={a.images}  out={a.out}  quick={a.quick}")
runpy.run_path(path, run_name="__main__")
