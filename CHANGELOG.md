# Changelog

## v1.1.0 (third-round revision)
- Learned detector repeated over three independent training seeds (`k6d_*` notebooks,
  `results/K6d_*`), 90 epochs, identical pairs and split across seeds, explicit
  convergence rule.
- `configs/`: effective full-mode constants for every notebook.
- `run_local.py` fixes `PYTHONHASHSEED=0` so local re-runs are repeatable.
- `k1-k5_revision_cpu` notebook set to `MODE="full"` / `REV_MODE="full"`, the mode in
  which it was run (the committed copy carried the proof-mode default; the result
  row counts in `results/K2_*`, `K3_*`, `K5_*` are the full-mode counts).
- Analysis scripts: AUC added to the sensitivity table; seeds table and figure.

## v1.0.0 (second-round revision)
- First public release: library and tests, notebooks K1–K9, local runner, results,
  table and figure scripts.
