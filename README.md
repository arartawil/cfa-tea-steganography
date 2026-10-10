# CFA-consistent image steganography with TEA/XTEA

Code, experiment notebooks and result files for

> L. Almazaydeh, A. Al Tawil, R. Al-Qudah, *Secure RGB Image Steganography Based on the
> Tiny Encryption Algorithm and Color Filter Array Consistency*, Computers, Materials &
> Continua (under review, submission 92460).

A single-sensor camera measures one colour per pixel through a Bayer colour filter
array and interpolates the other two. A real photograph therefore satisfies
`y = D(M(y))`, where `M` takes the Bayer mosaic and `D` is the demosaicing kernel.
Writing payload bits into the developed image breaks that relation. This scheme writes
them into the mosaic instead and develops again, so the stego image satisfies the
relation exactly.

The payload is encrypted with TEA or XTEA in counter mode. The same keystream supplies
the carrier order and the ±1 modification signs. Hamming syndrome coding reduces the
number of changes.

**Scope.** The guarantee holds for a known linear demosaicing kernel and a lossless
channel. It was evaluated on ALASKA#2 images re-mosaicked and re-developed with known
kernels, not on raw camera output. Passing the consistency test is not a general
guarantee of undetectability: a SPAM-686 detector reaches P_E = 0.325 at 0.20 bits per
pixel-channel. Any JPEG re-compression, noise or cropping destroys the payload. See the
paper, Section 7.

---

## Contents

```
cfa_stego/          the scheme as a small importable library (extracted verbatim
                    from experiments/study_core.py, so the two cannot diverge)
tests/              round-trip, consistency and wrong-key tests (pytest)
examples/hide_text.py   hide a sentence in an image, read it back, show what breaks it
experiments/
  study_core.py           the study script every notebook is built on
  k8_ablation_sensitivity.py, k9_cost.py   round-two experiment code
  notebooks/              the Kaggle notebooks exactly as they were run for the paper
  notebooks/superseded/   two failed learned-detector runs, kept for the record
  run_local.py            run any notebook on your own machine
results/            every CSV the paper's numbers come from (CC BY 4.0), see results/README.md
analysis/           mktables.py and mkfigs.py: regenerate every table and figure from results/
```

## Requirements

Python 3.10 or later (the paper's runs used 3.13). CPU only, except the learned
detector, which needs PyTorch and preferably a GPU.

```
pip install -r requirements.txt          # numpy, scipy, pandas, scikit-image, pillow, matplotlib, pytest
pip install -r requirements-gpu.txt      # adds torch, only for k6c
```

No other configuration is needed. All seeds are fixed in the code: the cover sample
seed is 20261003, the ten detector splits use seeds 1000 to 1009, and the learned
detector uses 2026.

## Quick start

```python
import numpy as np
from PIL import Image
import cfa_stego as cs

img = np.array(Image.open("photo.png").convert("RGB"))[:256, :256]
key = cs.key_from_passphrase("my secret")
msg = b"Meet at the north gate at 21:40."

stego, cover, info = cs.hide(img, msg, key, nonce=1, kernel="d4", cipher="xtea")
Image.fromarray(stego).save("stego.png")          # PNG only: the channel must be lossless

back = cs.reveal(np.array(Image.open("stego.png")), len(msg), key, nonce=1)
assert back == msg
print(cs.cfa_violation(stego, "d4"))              # 0.0: the stego image satisfies y = D(M(y))
```

The nonce must be unique for every image embedded under one key. The capacity of the
binary alphabet is one bit per mosaic sample, about 1/3 bit per pixel-channel
(`cs.capacity_bytes(img.shape)`).

Run the tests:

```
pytest -q tests
```

They check, for kernels d1, d2 and d4, both ciphers and three payloads, that the
payload comes back with zero bit errors and that cover and stego both satisfy the
consistency relation exactly. They also check that a wrong key recovers nothing and that
developed-domain LSB writes break the relation.

## Data

The covers are from ALASKA#2, distributed through the Kaggle competition
*ALASKA2 Image Steganalysis*
(<https://www.kaggle.com/competitions/alaska2-image-steganalysis>). You need a Kaggle
account and must accept the competition rules to download it. Only the `Cover/` folder
(75,000 JPEGs, 512×512) is used. No other preparation is needed. The code draws a seeded
random sample, centre-crops each image to 256×256 at an even origin (an odd origin would
swap the red and blue lattices), and develops it with the chosen kernel before use.

## Reproducing the results

Each result in the paper comes from one notebook. The notebooks run unchanged on
Kaggle, with the ALASKA#2 competition data attached and the GPU off except for k6c.
To run one locally instead:

```
python experiments/run_local.py --notebook experiments/notebooks/<name>.ipynb \
       --images /path/to/alaska2/Cover --out runs/<name>
```

Add `--quick` for a small smoke run in proof mode, which takes minutes rather than
hours.

| Paper result | Notebook | Output | Approx. time (4-core CPU) |
|---|---|---|---|
| Correctness, distortion, violation (Tables 3–6, Figs 2–3); robustness (Table 15) | `k7_reseeded_distortion_robustness` | `A_*`, `C_*` | 3–4 h |
| Detectability over ten splits, Wilcoxon (Tables 7–9, Fig. 4) | `k1-k5_revision_cpu` (RUN_K="1") + `k1b_payloads_010_040` | `K1_*`, `K1b_*` | 2–3 h each |
| Learned steganalysis (Tables 2 and 10, Fig. 5) | `k6c_learned_steganalysis_gpu` | `K6c_*` | ~3 h on one GPU |
| One-factor ablation (Tables 11–12, Fig. 6) | `k8a_one_factor_ablation` | `K8a_*` | ~4 h |
| Sensitivity (Table 13, Fig. 7): distortion columns | `k1-k5_revision_cpu` (RUN_K="3") | `K3_*` | ~1 h |
| Sensitivity (Table 13, Fig. 7): P_E columns | `k8b_sensitivity_steganalysis` | `K8b_*` | ~2.5 h |
| Kernel mismatch (Table 14, Fig. 8) | `k1-k5_revision_cpu` (RUN_K="4") | `K4_*` | < 1 h |
| Cost and hardware (Table 16) | `k9_hardware_cost_comparison` | `K9_*` | ~20 min |

Table and figure numbers refer to the second-round revision of the paper.

Every notebook writes its CSV incrementally, so an interrupted run keeps everything it
finished. To rebuild the tables and figures from `results/`:

```
python analysis/mktables.py        # LaTeX tables -> analysis/tables/
python analysis/mkfigs.py          # vector figures -> analysis/figures/
```

## Hardware used for the paper

All CPU experiments ran on Kaggle CPU notebooks: Intel Xeon @ 2.20 GHz, 4 logical
cores (AVX2), 32 GB RAM, Linux x86-64, Python 3.13.15, NumPy 2.1.3 (OpenBLAS 0.3.27),
SciPy 1.16.3, Pillow 12.3.0, scikit-image 0.25.2 (recorded in
`results/K9_environment.json`). The cost measurements pin every thread pool to one
thread. The learned detector ran in one Kaggle GPU session with PyTorch.

## What the experiments measure, and how

* **Distortion.** PSNR and SSIM against the developed cover. Change rate is the
  fraction of carriers modified, and efficiency is payload bits per change.
* **Correctness.** Bit error rate of the recovered payload on a lossless channel.
* **Structural consistency.** The fraction of interpolated sites in the image interior
  where `y != D(M(y))`.
* **Detectability.** Two trained detectors, a 23-feature consistency-residual detector
  and SPAM-686 per channel (2058 features), each with a Fisher-linear-discriminant
  ensemble. Both are evaluated over ten pair-preserving 500/500 splits of 1000
  cover–stego pairs, reporting P_E = min (P_FA + P_MD)/2 with 95% intervals and paired
  Wilcoxon tests.
* **Learned detector.** An SRM-initialised CNN after Ye et al. (2017), with its full
  configuration in the paper's CNN configuration table and in the k6c notebook. Its
  results are inconclusive, and the paper says so.

## Licence

Code: MIT (see `LICENSE`). Result files in `results/`: CC BY 4.0. The ALASKA#2 images
are not redistributed here and remain under the competition's terms.

## Citation

See `CITATION.cff`. Questions: Arar Al Tawil, or the corresponding author Laiali
Almazaydeh, via the contact details in the paper.
