import numpy as np, pytest
from scipy.ndimage import gaussian_filter
import cfa_stego as cs

def _scene(seed=0, n=128):
    r = np.random.default_rng(seed)
    a = np.stack([gaussian_filter(r.random((n, n)), s) * 400 - 60 for s in (2, 3, 4)], -1)
    return np.clip(a, 0, 255).astype(np.uint8)

@pytest.mark.parametrize("kernel", ["d1", "d2", "d4"])
@pytest.mark.parametrize("cipher", ["xtea", "tea"])
@pytest.mark.parametrize("bpc", [0.05, 0.10, 0.20])
def test_roundtrip_and_consistency(kernel, cipher, bpc):
    img = _scene()
    key = cs.key_from_passphrase("test")
    nb = int(round(bpc * img.size)) // 8
    msg = np.random.default_rng(1).integers(0, 256, nb, dtype=np.uint8).tobytes()
    stego, cover, info = cs.hide(img, msg, key, nonce=7, kernel=kernel, cipher=cipher)
    assert cs.reveal(stego, nb, key, 7, cipher) == msg          # BER = 0
    assert cs.cfa_violation(stego, kernel) == 0.0                 # y = D(M(y))
    assert cs.cfa_violation(cover, kernel) == 0.0

def test_wrong_key_recovers_nothing():
    img = _scene(2); msg = b"Meet at the north gate at 21:40." * 4
    stego, _, _ = cs.hide(img, msg, cs.key_from_passphrase("right"), 11)
    got = cs.reveal(stego, len(msg), cs.key_from_passphrase("wrong"), 11)
    assert got != msg

def test_developed_domain_lsb_breaks_consistency():
    img = cs.develop(_scene(3), "d4").astype(np.int16)
    img[::7, ::5, 1] ^= 1
    assert cs.cfa_violation(np.clip(img, 0, 255).astype(np.uint8), "d4") > 0
