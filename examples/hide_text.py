# =============================================================================
#  HIDE A TEXT MESSAGE IN AN IMAGE  --  end-to-end demonstration
#  ---------------------------------------------------------------------------
#  Type a sentence below, run the file once, and it will:
#
#      1. develop the cover so it lies on the CFA manifold
#      2. encrypt the text with TEA or XTEA in counter mode
#      3. embed the ciphertext into the SENSOR samples with Hamming coding
#      4. write stego.png to disk
#      5. read that PNG back from disk and recover the text
#      6. report PSNR, change rate, CFA violation rate and BER
#      7. show that the wrong key recovers nothing
#      8. show that JPEG destroys the payload -- the scope limit, demonstrated
#
#  Nothing is simulated: the message really is encrypted, really is embedded,
#  and really is read back off the disk.
# =============================================================================

# ---------------------------------------------------------------- settings --
MESSAGE = "Meet at the north gate at 21:40. Bring the blue folder."

IMAGE_IN  = None          # path to a cover image; None generates a test scene
STEGO_OUT = "stego.png"

PASSPHRASE = "cfa-stego-demo-2026"   # any string; the 128-bit key is its SHA-256
CIPHER     = "xtea"                  # "xtea" (recommended) or "tea"
KERNEL     = "d4"                    # "d1" | "d2" | "d4"
NONCE      = 20260903                # any integer, unique per image per key
SIZE       = 256                     # cover is centre-cropped to SIZE x SIZE

import os, sys, math, io, hashlib, warnings
import numpy as np
from scipy.signal import convolve2d
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ============================================================ CFA pipeline ==
# Bayer RGGB.  Reflect padding is used everywhere because it preserves the
# Bayer parity of the lattice -- that is what makes extraction exact.

def _cp(a, k, r):
    # r MUST be even: reflect padding then preserves the Bayer parity of the
    # lattice, which is what makes extraction exact rather than approximate.
    assert r % 2 == 0, "pad must be even to preserve Bayer parity"
    p = np.pad(a, r, mode="reflect")
    return convolve2d(p, k, mode="same")[r:-r, r:-r]

def bayer_mosaic(rgb):
    x = rgb.astype(np.float64); h, w = x.shape[:2]
    m = np.zeros((h, w))
    m[0::2, 0::2] = x[0::2, 0::2, 0]      # R
    m[0::2, 1::2] = x[0::2, 1::2, 1]      # G
    m[1::2, 0::2] = x[1::2, 0::2, 1]      # G
    m[1::2, 1::2] = x[1::2, 1::2, 2]      # B
    return m

def _planes(m):
    h, w = m.shape
    R = np.zeros((h, w)); G = np.zeros((h, w)); B = np.zeros((h, w))
    R[0::2, 0::2] = m[0::2, 0::2]
    G[0::2, 1::2] = m[0::2, 1::2]; G[1::2, 0::2] = m[1::2, 0::2]
    B[1::2, 1::2] = m[1::2, 1::2]
    return R, G, B

# --- d1 : bilinear, 3x3 support -------------------------------------------
K_RB1 = np.array([[.25, .5, .25], [.5, 1., .5], [.25, .5, .25]])
K_G1  = np.array([[0, .25, 0], [.25, 1., .25], [0, .25, 0]])

def demosaic_d1(m):
    R, G, B = _planes(m)
    out = np.stack([_cp(R, K_RB1, 2), _cp(G, K_G1, 2), _cp(B, K_RB1, 2)], -1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

# --- d2 : Malvar-He-Cutler, gradient-corrected, 5x5 support ---------------
# The MHC filters are applied to the FULL mosaic and then selected by Bayer
# parity, which is the standard vectorised form.
def _mhc():
    G = np.zeros((5, 5))                      # G at an R or B site
    G[2, 2] = 4; G[1, 2] = G[3, 2] = G[2, 1] = G[2, 3] = 2
    G[0, 2] = G[4, 2] = G[2, 0] = G[2, 4] = -1

    Hrow = np.zeros((5, 5))                   # R at G in an R row
    Hrow[2, 2] = 5; Hrow[2, 1] = Hrow[2, 3] = 4
    Hrow[2, 0] = Hrow[2, 4] = -1
    Hrow[1, 1] = Hrow[1, 3] = Hrow[3, 1] = Hrow[3, 3] = -1
    Hrow[0, 2] = Hrow[4, 2] = 0.5

    Hcol = Hrow.T.copy()                      # R at G in a B row

    D = np.zeros((5, 5))                      # R at a B site (and B at R)
    D[2, 2] = 6
    D[1, 1] = D[1, 3] = D[3, 1] = D[3, 3] = 2
    D[0, 2] = D[4, 2] = D[2, 0] = D[2, 4] = -1.5
    return G / 8.0, Hrow / 8.0, Hcol / 8.0, D / 8.0

K_MG, K_MHROW, K_MHCOL, K_MD = _mhc()

def demosaic_d2(m):
    h, w = m.shape
    fG    = _cp(m, K_MG,    4)
    fHrow = _cp(m, K_MHROW, 4)
    fHcol = _cp(m, K_MHCOL, 4)
    fD    = _cp(m, K_MD,    4)

    yy, xx = np.mgrid[0:h, 0:w]
    at_R  = (yy % 2 == 0) & (xx % 2 == 0)     # RGGB
    at_Gr = (yy % 2 == 0) & (xx % 2 == 1)     # G in an R row
    at_Gb = (yy % 2 == 1) & (xx % 2 == 0)     # G in a B row
    at_B  = (yy % 2 == 1) & (xx % 2 == 1)

    R = np.where(at_R, m, np.where(at_Gr, fHrow,
        np.where(at_Gb, fHcol, fD)))
    G = np.where(at_Gr | at_Gb, m, fG)
    B = np.where(at_B, m, np.where(at_Gb, fHrow,
        np.where(at_Gr, fHcol, fD)))

    out = np.stack([R, G, B], -1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

# --- d4 : wide 9x9 support -------------------------------------------------
def _wide(r=4):
    y, x = np.mgrid[-r:r+1, -r:r+1]
    k = np.exp(-(x**2 + y**2) / (2 * (r / 2.0) ** 2))
    return k / k.sum()

K_W = _wide(4)

def demosaic_d4(m):
    R, G, B = _planes(m)
    nR = np.zeros_like(m); nR[0::2, 0::2] = 1
    nG = np.zeros_like(m); nG[0::2, 1::2] = 1; nG[1::2, 0::2] = 1
    nB = np.zeros_like(m); nB[1::2, 1::2] = 1
    eps = 1e-9
    out = np.stack([_cp(R, K_W, 6) / (_cp(nR, K_W, 6) + eps),
                    _cp(G, K_W, 6) / (_cp(nG, K_W, 6) + eps),
                    _cp(B, K_W, 6) / (_cp(nB, K_W, 6) + eps)], -1)
    # measured sites are never overwritten by the interpolator
    out[0::2, 0::2, 0] = m[0::2, 0::2]
    out[0::2, 1::2, 1] = m[0::2, 1::2]
    out[1::2, 0::2, 1] = m[1::2, 0::2]
    out[1::2, 1::2, 2] = m[1::2, 1::2]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)

DEMOSAIC = {"d1": demosaic_d1, "d2": demosaic_d2, "d4": demosaic_d4}

def develop(rgb, d):
    return DEMOSAIC[d](bayer_mosaic(rgb))

U32 = np.uint32
DELTA = U32(0x9E3779B9)

# ================================================================== XTEA ====
def _as_key(key_bytes):
    if len(key_bytes) != 16:
        raise ValueError("key must be 16 bytes (128 bit)")
    return np.frombuffer(key_bytes, dtype=">u4").astype(U32)

def xtea_encrypt_blocks(v0, v1, key, rounds=32):
    """Vectorised XTEA. v0, v1 are uint32 arrays of equal length."""
    v0 = v0.astype(U32).copy(); v1 = v1.astype(U32).copy()
    s = U32(0)
    with np.errstate(over="ignore"):
        for _ in range(rounds):
            v0 += (((v1 << U32(4)) ^ (v1 >> U32(5))) + v1) ^ (s + key[s & U32(3)])
            s = U32(s + DELTA)
            v1 += (((v0 << U32(4)) ^ (v0 >> U32(5))) + v0) ^ \
                  (s + key[(s >> U32(11)) & U32(3)])
    return v0, v1

def xtea_decrypt_blocks(v0, v1, key, rounds=32):
    v0 = v0.astype(U32).copy(); v1 = v1.astype(U32).copy()
    s = U32((int(DELTA) * rounds) & 0xFFFFFFFF)
    with np.errstate(over="ignore"):
        for _ in range(rounds):
            v1 -= (((v0 << U32(4)) ^ (v0 >> U32(5))) + v0) ^ \
                  (s + key[(s >> U32(11)) & U32(3)])
            s = U32(s - DELTA)
            v0 -= (((v1 << U32(4)) ^ (v1 >> U32(5))) + v1) ^ (s + key[s & U32(3)])
    return v0, v1

# ============================================================ original TEA ==
#  The grant is titled "... based on Tiny Encryption Algorithm", so TEA is
#  not a footnote here -- it is a deliverable.  Both ciphers are therefore
#  implemented and BOTH are run as full arms of the experiment.
#
#  Why XTEA is nevertheless the recommendation: in TEA every key has three
#  equivalent keys that produce identical ciphertext, so the effective key
#  is 126 bits rather than 128.  That is the weakness exploited in the first
#  Xbox hack, where TEA was used as a hash.  XTEA is the same two authors'
#  repair of exactly this flaw.  Running both lets the manuscript SHOW that
#  the choice is purely a cryptographic one and does not touch the
#  steganographic result, instead of merely asserting it.

def tea_encrypt_blocks(v0, v1, key, rounds=32):
    """Vectorised original TEA (Wheeler & Needham, 1994)."""
    v0 = v0.astype(U32).copy(); v1 = v1.astype(U32).copy()
    s = U32(0)
    with np.errstate(over="ignore"):
        for _ in range(rounds):
            s = U32(s + DELTA)
            v0 += (((v1 << U32(4)) + key[0]) ^ (v1 + s) ^
                   ((v1 >> U32(5)) + key[1]))
            v1 += (((v0 << U32(4)) + key[2]) ^ (v0 + s) ^
                   ((v0 >> U32(5)) + key[3]))
    return v0, v1

def tea_decrypt_blocks(v0, v1, key, rounds=32):
    v0 = v0.astype(U32).copy(); v1 = v1.astype(U32).copy()
    s = U32((int(DELTA) * rounds) & 0xFFFFFFFF)
    with np.errstate(over="ignore"):
        for _ in range(rounds):
            v1 -= (((v0 << U32(4)) + key[2]) ^ (v0 + s) ^
                   ((v0 >> U32(5)) + key[3]))
            v0 -= (((v1 << U32(4)) + key[0]) ^ (v1 + s) ^
                   ((v1 >> U32(5)) + key[1]))
            s = U32(s - DELTA)
    return v0, v1

CIPHERS = {"xtea": (xtea_encrypt_blocks, xtea_decrypt_blocks),
           "tea":  (tea_encrypt_blocks,  tea_decrypt_blocks)}

def ctr_keystream(key_bytes, nonce, n_words, cipher="xtea"):
    """Counter-mode keystream: n_words uint32 values, from either cipher."""
    key = _as_key(key_bytes)
    n_blocks = (n_words + 1) // 2
    v0 = np.arange(n_blocks, dtype=U32)
    v1 = np.full(n_blocks, U32(nonce & 0xFFFFFFFF), dtype=U32)
    c0, c1 = CIPHERS[cipher][0](v0, v1, key)
    out = np.empty(n_blocks * 2, dtype=U32)
    out[0::2] = c0; out[1::2] = c1
    return out[:n_words]

def xtea_ctr_keystream(key_bytes, nonce, n_words):   # back-compat alias
    return ctr_keystream(key_bytes, nonce, n_words, "xtea")

def _keystream_bytes(key_bytes, nonce, n_bytes, cipher="xtea"):
    w = ctr_keystream(key_bytes, nonce, (n_bytes + 3) // 4, cipher)
    return w.view(np.uint8)[:n_bytes] if w.dtype.byteorder in "=<" \
        else w.astype("<u4").view(np.uint8)[:n_bytes]

# ================================================= keyed derived material ===
def _bits_of(byte_arr):
    return np.unpackbits(np.asarray(byte_arr, dtype=np.uint8))

def encrypt_payload(msg_bytes, key_bytes, nonce, cipher="xtea"):
    ks = _keystream_bytes(key_bytes, nonce ^ 0xA5A5A5A5, len(msg_bytes), cipher)
    return np.frombuffer(msg_bytes, dtype=np.uint8) ^ ks

def decrypt_payload(cipher_arr, key_bytes, nonce, cipher="xtea"):
    ks = _keystream_bytes(key_bytes, nonce ^ 0xA5A5A5A5, len(cipher_arr), cipher)
    return (np.asarray(cipher_arr, dtype=np.uint8) ^ ks).tobytes()

def keyed_permutation(n, key_bytes, nonce, cipher="xtea"):
    """Deterministic keyed ordering of the n mosaic carriers."""
    w = ctr_keystream(key_bytes, nonce ^ 0x5A5A5A5A, n, cipher)
    return np.argsort(w, kind="stable")

def keyed_signs(n, key_bytes, nonce, cipher="xtea"):
    w = ctr_keystream(key_bytes, nonce ^ 0x3C3C3C3C, n, cipher)
    return np.where((w & U32(1)) == 0, 1, -1).astype(np.int16)

# ============================================== Hamming syndrome coding =====
def _H(p):
    """Parity-check matrix: columns are the binary forms of 1 .. 2^p - 1."""
    n = (1 << p) - 1
    cols = np.arange(1, n + 1, dtype=np.uint32)
    return ((cols[None, :] >> np.arange(p, dtype=np.uint32)[:, None]) & 1
            ).astype(np.uint8)

def choose_p(rate_needed, p_max=8):
    """Largest p whose code rate p/(2^p - 1) still meets the payload."""
    best = None
    for p in range(1, p_max + 1):
        if p / ((1 << p) - 1) >= rate_needed:
            best = p
    if best is None:
        raise ValueError(
            f"payload needs {rate_needed:.3f} bits per carrier; "
            f"Hamming coding caps at 1.0 (p=1)")
    return best

def hamming_embed(lsb, msg_bits, p):
    """lsb: (n_blocks, 2^p-1) uint8. msg_bits: (n_blocks, p) uint8.
    Returns the index to flip in each block, or -1 for no change."""
    H = _H(p)
    syn = (lsb @ H.T) & 1                       # (n_blocks, p)
    d = (syn ^ msg_bits).astype(np.uint32)      # required syndrome change
    pos = (d * (1 << np.arange(p, dtype=np.uint32))[None, :]).sum(1)
    return np.where(pos == 0, -1, pos.astype(np.int64) - 1)

def hamming_extract(lsb, p):
    H = _H(p)
    return (lsb @ H.T) & 1



# ------------- keyed XTEA + Hamming scheme, any demosaic kernel ------------
def cfa_embed(cover_rgb, msg_bytes, key_bytes, demos_fn, nonce=0,
              cipher="xtea"):
    m = bayer_mosaic(cover_rgb).astype(np.int16)
    n_carriers = m.size
    ct = encrypt_payload(msg_bytes, key_bytes, nonce, cipher)
    bits = _bits_of(ct)
    n_bits = bits.size

    p = choose_p(n_bits / n_carriers)
    blk = (1 << p) - 1
    n_blocks = int(np.ceil(n_bits / p))
    need = n_blocks * blk
    if need > n_carriers:
        raise ValueError("payload exceeds carriers")

    bits_pad = np.zeros(n_blocks * p, np.uint8); bits_pad[:n_bits] = bits
    perm = keyed_permutation(n_carriers, key_bytes, nonce, cipher)
    sites = perm[:need].reshape(n_blocks, blk)

    flat = m.ravel().copy()
    lsb = (flat[sites] & 1).astype(np.uint8)
    flip = hamming_embed(lsb, bits_pad.reshape(n_blocks, p), p)

    rows = np.flatnonzero(flip >= 0)
    if rows.size:
        tgt = sites[rows, flip[rows]]
        signs = keyed_signs(n_carriers, key_bytes, nonce, cipher)[tgt]
        vals = flat[tgt]
        signs = np.where((vals <= 0) & (signs < 0), 1, signs)
        signs = np.where((vals >= 255) & (signs > 0), -1, signs)
        flat[tgt] = vals + signs
    m2 = np.clip(flat.reshape(m.shape), 0, 255).astype(np.float64)
    return demos_fn(m2), dict(p=p, change_rate=float(rows.size) / n_carriers,
                              efficiency=float(n_bits) / max(rows.size, 1))

def cfa_extract(stego_rgb, n_msg_bytes, key_bytes, nonce=0,
                cipher="xtea"):
    m = bayer_mosaic(stego_rgb).astype(np.int16)
    n_carriers = m.size
    n_bits = n_msg_bytes * 8
    p = choose_p(n_bits / n_carriers)
    blk = (1 << p) - 1
    n_blocks = int(np.ceil(n_bits / p))
    perm = keyed_permutation(n_carriers, key_bytes, nonce, cipher)
    sites = perm[:n_blocks * blk].reshape(n_blocks, blk)
    lsb = (m.ravel()[sites] & 1).astype(np.uint8)
    bits = hamming_extract(lsb, p).reshape(-1)[:n_bits]
    return decrypt_payload(np.packbits(bits)[:n_msg_bytes], key_bytes,
                           nonce, cipher)

def xtea_embed(*a, **k):    return cfa_embed(*a, **k)      # back-compat
def xtea_extract(*a, **k):  return cfa_extract(*a, **k)

# ================================================================== costs ==
HILL_DYN = 1e4          # bound on the cost dynamic range (see below)

def hill_cost(plane, dyn=HILL_DYN):
    """HILL: high-pass residual, smoothed twice.

    FIX (v2).  The textbook form inverts (s + 1e-10).  On a genuinely smooth
    region -- sky, a wall, a blown highlight -- the smoothed residual s is
    numerically zero, so the cost reaches 1e10 and exp(-lambda*rho)
    underflows for every lambda the bisection can reach.  Those samples are
    then PERMANENTLY WET: the payload-distortion simulator cannot place a bit
    on them, and on a smooth crop the method silently carries far less than
    the requested payload while still being scored as if it carried all of
    it.  That is a fabricated advantage, and it is what the "payload gap"
    check catches.

    Flooring s relative to its own mean keeps the cost fully content
    adaptive but bounds its dynamic range to `dyn`, so no sample is ever
    unreachable and the requested payload is actually carried."""
    H = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], float)
    L1 = np.ones((3, 3)) / 9.0
    L2 = np.ones((15, 15)) / 225.0
    r = np.abs(convolve2d(plane.astype(float), H, mode="same", boundary="symm"))
    s = convolve2d(r, L1, mode="same", boundary="symm")
    s = np.maximum(s, (float(s.mean()) + 1e-12) / dyn)
    s = convolve2d(1.0 / s, L2, mode="same", boundary="symm")
    return np.clip(s, 1e-12, 1e12)

# ============================================== payload-distortion bound ===
def _h3(p):
    p = np.clip(p, 1e-12, 1 / 3 - 1e-12)
    return -2 * p * np.log2(p) - (1 - 2 * p) * np.log2(1 - 2 * p)

def _p_of_lam(rho, lam):
    e = np.exp(-lam * rho)
    return e / (1 + 2 * e)

def simulate_at_bound(rho, n_bits, rng):
    """Ternary embedding simulator at the payload-distortion bound.
    lambda by bisection on the ternary entropy.

    FIX (v2).  rho is normalised by its own mean before the bisection, so
    the bracket is scale free; with an un-normalised rho the fixed bracket
    [1e-6, 1e6] could sit entirely on one side of the solution and the
    search would return the endpoint rather than the root."""
    rn = np.asarray(rho, float)
    rn = rn / (float(rn.mean()) + 1e-12)
    lo, hi = 1e-8, 1e8
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if _h3(_p_of_lam(rn, mid)).sum() > n_bits:
            lo = mid
        else:
            hi = mid
    p = _p_of_lam(rn, math.sqrt(lo * hi))
    u = rng.random(rho.shape)
    d = np.zeros(rho.shape, np.int16)
    d[u < p] = 1
    d[(u >= p) & (u < 2 * p)] = -1
    achieved = float(_h3(p).sum())
    return d, achieved


# ============================================================== the demo ====
def short(t, n=88):
    t = t if isinstance(t, str) else repr(t)
    return t if len(t) <= n else t[:n-3] + "..."

def _rule(t=""):
    print("=" * 74)
    if t: print(t); print("=" * 74)

def make_scene(n):
    """A synthetic photograph, used when no cover image is supplied."""
    y, x = np.mgrid[0:n, 0:n].astype(float)
    u, v = x / n, y / n
    r = 120 + 70*np.sin(6.0*u + 1.2) + 28*np.cos(4.4*v)
    g = 118 + 62*np.sin(5.4*u + 0.4) + 34*np.cos(3.6*v + 1.1)
    b = 132 + 46*np.sin(4.6*u - 0.7) + 30*np.cos(5.2*v + 0.3)
    disc = (x-0.33*n)**2 + (y-0.41*n)**2 < (0.19*n)**2
    r = np.where(disc, r+54, r); g = np.where(disc, g-26, g)
    b = np.where(disc, b-40, b)
    band = y > 0.70*n
    t = 16*np.sin(x*1.9)*np.cos(y*1.3)
    r = np.where(band, r+t, r); g = np.where(band, g+0.8*t, g)
    b = np.where(band, b+0.5*t, b)
    return np.clip(np.stack([r, g, b], -1), 0, 255).astype(np.uint8)

def load_cover(path, size):
    from PIL import Image
    if path is None:
        return make_scene(size), "synthetic test scene"
    a = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    h, w = a.shape[:2]
    if h < size or w < size:
        raise SystemExit(f"STOP: {path} is {w}x{h}, smaller than SIZE={size}.")
    # an EVEN crop origin -- an odd one shifts the Bayer phase and swaps R and B
    y0, x0 = ((h-size)//2) & ~1, ((w-size)//2) & ~1
    return a[y0:y0+size, x0:x0+size], os.path.basename(path)

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64))**2)
    return float("inf") if mse == 0 else float(10*np.log10(255.0**2/mse))

def cfa_violation(img, d):
    re = DEMOSAIC[d](bayer_mosaic(img))
    diff = img.astype(np.int16) != re.astype(np.int16)
    h, w = img.shape[:2]
    par = np.add.outer(np.arange(h) % 2, np.arange(w) % 2)
    interp = np.stack([par != 0, par != 1, par != 2], -1)
    inner = np.zeros((h, w), bool); inner[6:-6, 6:-6] = True
    sel = interp & inner[:, :, None]
    return float(diff[sel].mean()) if sel.any() else float("nan")

def ber(a, b):
    """Bit error rate, as reported in the paper -- not a byte mismatch rate."""
    n = min(len(a), len(b))
    if n == 0: return 1.0
    pa = np.unpackbits(np.frombuffer(a[:n], np.uint8))
    pb = np.unpackbits(np.frombuffer(b[:n], np.uint8))
    return float(np.mean(pa != pb))

def main():
    from PIL import Image
    key = hashlib.sha256(PASSPHRASE.encode()).digest()[:16]
    msg = MESSAGE.encode("utf-8")

    _rule("HIDE A TEXT MESSAGE IN AN IMAGE")
    raw, srcname = load_cover(IMAGE_IN, SIZE)
    cover = develop(raw, KERNEL)          # put the cover on the CFA manifold
    n_car = cover.shape[0]*cover.shape[1]

    print(f"  cover            : {srcname}  {cover.shape[1]}x{cover.shape[0]}")
    print(f"  cipher / kernel  : {CIPHER.upper()} / {KERNEL}")
    print(f"  message          : \"{short(MESSAGE)}\"")
    print(f"  message size     : {len(msg)} bytes = {len(msg)*8} bits")
    print(f"  carriers         : {n_car} mosaic samples "
          f"({n_car/(cover.size):.0%} of the image samples)")
    print(f"  payload rate     : {len(msg)*8/cover.size:.5f} bits per pixel-channel "
          f"(capacity {1/3:.4f})")
    v = cfa_violation(cover, KERNEL)
    print(f"  cover CFA check  : violation rate {v:.1e}  "
          f"{'OK' if v < 1e-12 else 'COVER NOT ON THE MANIFOLD'}")

    if len(msg)*8 > n_car:
        raise SystemExit("STOP: message longer than the carrier capacity.")

    # ------------------------------------------------------------- embed ---
    stego, info = cfa_embed(cover, msg, key, DEMOSAIC[KERNEL], NONCE, CIPHER)
    Image.fromarray(stego).save(STEGO_OUT)

    print()
    _rule("EMBEDDED")
    print(f"  Hamming p        : {info['p']}  "
          f"(block length {2**info['p']-1}, {info['efficiency']:.2f} bits per change)")
    print(f"  samples changed  : {info['change_rate']*n_car:.0f} "
          f"of {n_car}  ({info['change_rate']:.4%})")
    print(f"  PSNR             : {psnr(cover, stego):.2f} dB")
    print(f"  max pixel change : "
          f"{int(np.abs(cover.astype(int)-stego.astype(int)).max())} "
          f"(out of 255)")
    print(f"  CFA violations   : {cfa_violation(stego, KERNEL):.1e}   "
          f"<-- the whole point")
    print(f"  written          : {STEGO_OUT} "
          f"({os.path.getsize(STEGO_OUT)/1024:.0f} KB)")

    # ---------------------------------------- extract, from the file itself -
    back_img = np.array(Image.open(STEGO_OUT).convert("RGB"), dtype=np.uint8)
    rec = cfa_extract(back_img, len(msg), key, NONCE, CIPHER)

    print()
    _rule("EXTRACTED  (read back from the PNG on disk)")
    try:
        text = rec.decode("utf-8")
    except UnicodeDecodeError:
        text = repr(rec)
    print(f"  recovered        : \"{short(text)}\"")
    print(f"  BER              : {ber(msg, rec):.6f}")
    ok = (rec == msg)
    print(f"  byte-exact       : {ok}   {'PASS' if ok else 'FAIL'}")

    # --------------------------------------------------- the wrong key ------
    wrong = hashlib.sha256(b"not-the-passphrase").digest()[:16]
    bad = cfa_extract(back_img, len(msg), wrong, NONCE, CIPHER)
    print()
    _rule("WRONG KEY")
    print(f"  recovered        : {bad[:44]!r}...")
    print(f"  BER vs message   : {ber(msg, bad):.4f}   "
          f"(0.5 is pure chance)")
    print("  The permutation, the signs and the payload cipher all derive from")
    print("  the key, so without it the extractor does not even know which")
    print("  samples to read.")

    # --------------------------------------------------- the scope limit ----
    buf = io.BytesIO()
    Image.fromarray(stego).save(buf, format="JPEG", quality=95)
    buf.seek(0)
    jpg = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    jrec = cfa_extract(jpg, len(msg), key, NONCE, CIPHER)
    print()
    _rule("AFTER JPEG (quality 95)")
    print(f"  recovered        : {jrec[:44]!r}...")
    print(f"  BER              : {ber(msg, jrec):.4f}")
    print("  The payload does not survive lossy re-compression. This scheme is")
    print("  for lossless channels -- PNG, TIFF, raw -- and is not a watermark.")

    print()
    _rule("WHAT THE RECIPIENT NEEDS")
    print(f"  passphrase       : {PASSPHRASE!r}")
    print(f"  cipher, kernel   : {CIPHER}, {KERNEL}")
    print(f"  nonce            : {NONCE}")
    print(f"  message length   : {len(msg)} bytes")
    print("  Nothing else. The stego image itself carries no header.")
    _rule()

if __name__ == "__main__":
    main()
