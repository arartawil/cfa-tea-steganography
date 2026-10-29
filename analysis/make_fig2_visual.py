"""Figure 2 of the paper (visual example).

Usage:  python analysis/make_fig2_visual.py /path/to/alaska2/Cover/00001.jpg analysis/figures/F6_visual_example.pdf

Extractable Proposed (XTEA) with d4 against LSB matching at 0.20 bits per pixel-channel on the
same developed cover (ALASKA#2 Cover/00001.jpg, even-origin 256x256 centre crop): the images,
the amplified differences from the cover, and the maps of interpolated sites violating y = D(M(y)).
The payload is extracted and checked before the figure is drawn.
"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cfa_stego as cs
img_path, out_pdf = sys.argv[1], sys.argv[2]
BPC, K, AMP = 0.20, "d4", 64
raw = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
h, w = raw.shape[:2]; s = 256
y, x = ((h - s) // 2) & ~1, ((w - s) // 2) & ~1          # even-origin centre crop, as in the study
raw = raw[y:y+s, x:x+s]
key = cs.key_from_passphrase("figure-2"); nonce = 7
nb = int(round(BPC * raw.size)) // 8
msg = np.random.default_rng(0).integers(0, 256, nb, dtype=np.uint8).tobytes()
stego, cover, info = cs.hide(raw, msg, key, nonce, kernel=K, cipher="xtea")
assert cs.reveal(stego, nb, key, nonce, "xtea") == msg
# LSB matching in the developed domain, same payload in bits
rng = np.random.default_rng(0); xx = cover.astype(np.int16).ravel().copy()
k = int(round(BPC * xx.size)); idx = rng.choice(xx.size, k, replace=False)
bits = rng.integers(0, 2, k); cur = xx[idx]; need = (cur & 1) != bits
step = np.where(rng.random(k) < 0.5, 1, -1); step = np.where(cur + step > 255, -1, np.where(cur + step < 0, 1, step))
xx[idx] = np.where(need, cur + step, cur); lsbm = np.clip(xx, 0, 255).astype(np.uint8).reshape(cover.shape)
def vmap(im):
    re = cs.DEMOSAIC[K](cs.bayer_mosaic(im))
    return (im.astype(np.int16) != re.astype(np.int16)).any(2)
def amp(a, b):
    return np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)).max(2) * AMP, 0, 255)
cols = [("Cover (developed with $d_4$)", cover, None),
        ("Proposed (XTEA), $d_4$, 0.20 bpc", stego, cover),
        ("LSB matching, 0.20 bpc", lsbm, cover)]
plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, ax = plt.subplots(3, 3, figsize=(8.4, 8.6))
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_color("#999999"); sp.set_linewidth(0.8)
for j, (ttl, im, ref) in enumerate(cols):
    ax[0, j].imshow(im)
    ax[0, j].set_title(ttl if ref is None else f"{ttl}\nPSNR {cs.psnr(ref, im):.2f} dB", fontsize=9.5)
    if ref is None:
        ax[1, j].imshow(np.zeros(im.shape[:2]), cmap="gray", vmin=0, vmax=255); ax[1, j].set_title("Difference (reference)", fontsize=9.5)
    else:
        ax[1, j].imshow(amp(im, ref), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax[1, j].set_title(f"|stego $-$ cover| $\\times${AMP}", fontsize=9.5)
    v = vmap(im)
    ax[2, j].imshow(v, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
    ax[2, j].set_title(f"Violation rate {cs.cfa_violation(im, K):.4f}", fontsize=9.5)
ax[0, 0].set_ylabel("Image", fontsize=10); ax[1, 0].set_ylabel("Amplified difference", fontsize=10); ax[2, 0].set_ylabel("Violations of $y=D(M(y))$", fontsize=10)
fig.tight_layout(); fig.savefig(out_pdf, bbox_inches="tight"); fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=110, bbox_inches="tight")
print(dict(psnr_prop=round(cs.psnr(cover, stego), 3), psnr_lsbm=round(cs.psnr(cover, lsbm), 3),
           viol_cover=cs.cfa_violation(cover, K), viol_prop=cs.cfa_violation(stego, K), viol_lsbm=round(cs.cfa_violation(lsbm, K), 4),
           change_rate=round(info["change_rate"], 4), payload_bytes=nb, ber=0.0))
