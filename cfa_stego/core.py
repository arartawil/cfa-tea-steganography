"""Core of the CFA-consistent TEA/XTEA steganography scheme.

Extracted verbatim from experiments/study_core.py, the script that produced every
number in the paper, so the library and the experiments cannot drift apart.

Pipeline: re-mosaic the cover (Bayer RGGB) -> encrypt the payload with TEA/XTEA in
counter mode -> Hamming syndrome coding into the mosaic samples at a keyed carrier
order with keyed +/-1 signs -> develop with the same linear kernel.  The stego image
satisfies y = D(M(y)) exactly.
"""
import hashlib
import numpy as np
from scipy.signal import convolve2d

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


K_RB1 = np.array([[.25, .5, .25], [.5, 1., .5], [.25, .5, .25]])


K_G1  = np.array([[0, .25, 0], [.25, 1., .25], [0, .25, 0]])


def demosaic_d1(m):
    R, G, B = _planes(m)
    out = np.stack([_cp(R, K_RB1, 2), _cp(G, K_G1, 2), _cp(B, K_RB1, 2)], -1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


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


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def cfa_violation(img, d):
    """The paper's own constraint: does the image satisfy x = D(M(x))?"""
    re = DEMOSAIC[d](bayer_mosaic(img))
    diff = img.astype(np.int16) != re.astype(np.int16)
    h, w = img.shape[:2]
    par = np.add.outer(np.arange(h) % 2, np.arange(w) % 2)
    interp = np.stack([par != 0, par != 1, par != 2], -1)
    inner = np.zeros((h, w), bool); inner[6:-6, 6:-6] = True
    sel = interp & inner[:, :, None]
    return float(diff[sel].mean()) if sel.any() else np.nan


# ------------------------------------------------------------ convenience API --
def key_from_passphrase(passphrase: str) -> bytes:
    """128-bit key = first 16 bytes of SHA-256(passphrase)."""
    return hashlib.sha256(passphrase.encode("utf-8")).digest()[:16]


def hide(cover_rgb, message: bytes, key: bytes, nonce: int, kernel: str = "d4",
         cipher: str = "xtea"):
    """Develop the cover with `kernel`, then embed `message`.

    Returns (stego, developed_cover, info).  `nonce` must be unique per image
    under one key.  `kernel` is one of "d1", "d2", "d4"; `cipher` "xtea" or "tea".
    """
    cover = develop(np.asarray(cover_rgb, np.uint8), kernel)
    stego, info = cfa_embed(cover, message, key, DEMOSAIC[kernel], nonce, cipher)
    return np.asarray(stego, np.uint8), cover, info


def reveal(stego_rgb, n_bytes: int, key: bytes, nonce: int, cipher: str = "xtea") -> bytes:
    """Recover `n_bytes` of payload. Extraction is kernel-independent."""
    return cfa_extract(np.asarray(stego_rgb, np.uint8), n_bytes, key, nonce, cipher)


def capacity_bytes(shape) -> int:
    """Largest payload (bytes) for an H x W x 3 image: one bit per mosaic sample
    is the binary-alphabet limit (1/3 bit per pixel-channel)."""
    h, w = shape[:2]
    return (h * w) // 8
