# =============================================================================
#  K8  --  ROUND 2, Reviewer 1 comments 1 and 4
#
#  K8a  one-factor-at-a-time ablation WITH steganalysis (SPAM-686 and the CFA
#       detector), every arm on the same kernel (d4) unless stated.
#  K8b  sensitivity (kernel support, Hamming p, carrier policy) WITH steganalysis.
#
#  Design rules that answer the reviewer:
#   1. each arm differs from the full system in exactly ONE component
#   2. the cipher is only meaningful when the plaintext is NOT already random,
#      so the cipher arms are run on random AND on structured (ASCII text)
#      payloads.  With a random payload the cipher is a statistical no-op and
#      the experiment could not show anything else.
#   3. the CFA-domain LSB-matching control is run on d4, not d1
#   4. every arm records PSNR, change rate, efficiency and BER per image, and
#      P_E / AUC per split, so every claim has an interval
# =============================================================================

TEXT_BLOB = (
    b"Steganography hides the existence of a message rather than its meaning. "
    b"A courier who carries a sealed letter reveals that a letter exists; a courier "
    b"who carries nothing but a photograph does not. The photograph in this study "
    b"is developed from a colour filter array mosaic, and every interpolated sample "
    b"is a fixed linear function of the measured samples around it. A message "
    b"written into the developed pixels breaks that relation, while a message "
    b"written into the measured samples and then developed again does not. "
    b"Dear colleague, please find enclosed the quarterly report, the revised "
    b"budget and the minutes of the last meeting. Kind regards. ")


def _payload_bytes(nbytes, kind, rng):
    if kind == "text":
        off = int(rng.integers(0, len(TEXT_BLOB)))
        rep = (nbytes // len(TEXT_BLOB)) + 2
        return (TEXT_BLOB * rep)[off:off + nbytes]
    return rng.integers(0, 256, nbytes, dtype=np.uint8).tobytes()


def embed_factorial(cover, kfn, bpc, rng, *, cipher_on=True, keyed_sites=True,
                    keyed_sign=True, coding=True, payload="random",
                    p_force=None, cost_aware=False):
    """The proposed scheme with each component independently switchable.

    cipher_on    payload is XORed with the TEA/XTEA keystream (else plaintext)
    keyed_sites  carrier order is the keyed permutation (else raster order)
    keyed_sign   modification sign comes from the keystream (else fixed +1)
    coding       Hamming syndrome coding (else one payload bit per carrier)
    cost_aware   carriers ordered by HILL cost of the COVER (not extractable
                 without side information; reported as an oracle)

    The keyed permutation and signs are always DERIVED from the same
    keystream, so switching the cipher off does not remove the keyed layout.
    """
    m = bayer_mosaic(cover).astype(np.int16)
    n_out, n_car = cover.size, m.size
    n_bits = int(round(bpc * n_out))
    nbytes = n_bits // 8
    if nbytes == 0:
        return None, dict(status="payload_too_small")
    nonce = int(rng.integers(0, 2**31))
    msg = _payload_bytes(nbytes, payload, rng)
    if cipher_on:
        bits = _bits_of(encrypt_payload(msg, XTEA_KEY, nonce, "xtea"))
    else:
        bits = np.unpackbits(np.frombuffer(msg, np.uint8))

    if coding:
        p = int(p_force) if p_force else choose_p(bits.size / n_car)
        if p < 1 or p > 8:
            return None, dict(status="bad_p")
        blk, n_blocks = (1 << p) - 1, int(np.ceil(bits.size / p))
        need = n_blocks * blk
    else:
        p, blk, n_blocks, need = 0, 1, bits.size, bits.size
    if need > n_car:
        return None, dict(status="capacity_exceeded")

    if cost_aware:
        order = np.argsort(hill_cost(bayer_mosaic(cover)).ravel(), kind="stable")
    elif keyed_sites:
        order = keyed_permutation(n_car, XTEA_KEY, nonce, "xtea")
    else:
        order = np.arange(n_car)
    sites = order[:need]
    sg_all = (keyed_signs(n_car, XTEA_KEY, nonce, "xtea") if keyed_sign
              else np.ones(n_car, np.int16))

    flat = m.ravel().copy()
    if coding:
        sites2 = sites.reshape(n_blocks, blk)
        bits_pad = np.zeros(n_blocks * p, np.uint8); bits_pad[:bits.size] = bits
        lsb = (flat[sites2] & 1).astype(np.uint8)
        flip = hamming_embed(lsb, bits_pad.reshape(n_blocks, p), p)
        rows = np.flatnonzero(flip >= 0)
        tgt = sites2[rows, flip[rows]] if rows.size else np.array([], np.int64)
        changed = rows.size
    else:
        cur = flat[sites]
        needs = (cur & 1) != bits
        tgt = sites[needs]
        changed = int(needs.sum())
    if tgt.size:
        vals = flat[tgt]
        sg = sg_all[tgt]
        sg = np.where((vals <= 0) & (sg < 0), 1, sg)
        sg = np.where((vals >= 255) & (sg > 0), -1, sg)
        flat[tgt] = vals + sg
    m2 = np.clip(flat.reshape(m.shape), 0, 255).astype(np.float64)
    stego = kfn(m2)

    rec = bayer_mosaic(stego).astype(np.int16).ravel()
    if coding:
        got = hamming_extract((rec[sites.reshape(n_blocks, blk)] & 1)
                              .astype(np.uint8), p).reshape(-1)[:bits.size]
    else:
        got = (rec[sites] & 1).astype(np.uint8)
    ber = float(np.mean(got != bits))
    return stego, dict(status="ok", p=p, change_rate=changed / n_car,
                       actual_bpc=bits.size / n_out, ber=ber,
                       efficiency=bits.size / max(changed, 1))


# --------------------------------------------------------------- arm tables
def _fx(**kw):
    return lambda c, kfn, b, r, _kw=kw: embed_factorial(c, kfn, b, r, **_kw)

def _naive_with(cover, kfn, bpc, rng):
    m = bayer_mosaic(cover).astype(np.int16)
    n_out, n_sites = cover.size, m.size
    k = int(round(bpc * n_out))
    if k > n_sites:
        return None, dict(status="capacity_exceeded")
    idx = rng.choice(n_sites, k, replace=False)
    flat = m.ravel().copy()
    bits = _rand_bits(k, rng)
    cur = flat[idx]
    need = (cur & 1) != bits
    step = np.where(rng.random(k) < 0.5, 1, -1).astype(np.int16)
    step = _clamp_delta(cur, step)
    flat[idx] = np.where(need, cur + step, cur)
    m2 = np.clip(flat.reshape(m.shape), 0, 255).astype(np.float64)
    stego = kfn(m2)
    rec = bayer_mosaic(stego).astype(np.int16).ravel()
    return stego, dict(status="ok", change_rate=float(np.mean(m2 != m)),
                       actual_bpc=k / n_out,
                       ber=float(np.mean((rec[idx] & 1) != bits)),
                       efficiency=float(k) / max(int((m2 != m).sum()), 1))

def _sim_arm(c, kfn, b, r):
    # kfn is d4's developer here; ours_sim takes the kernel NAME
    st, info = ours_sim(c, "d4", b, r)
    if st is None:
        return None, dict(status=info.get("status", "fail"))
    return st, dict(status="ok", change_rate=info["change_rate"],
                    actual_bpc=info["actual_bpc"], ber=info["ber"],
                    efficiency=np.nan)

K8A_ARMS = [   # name, fn, payload kinds, extractable?
    ("full",            _fx(),                                   ("random", "text"), True),
    ("minus_cipher",    _fx(cipher_on=False),                    ("random", "text"), True),
    ("minus_keyed_sites", _fx(keyed_sites=False),                ("random",),        True),
    ("minus_keyed_sign",  _fx(keyed_sign=False),                 ("random",),        True),
    ("minus_coding",    _fx(coding=False),                       ("random",),        True),
    ("minus_all_three", _fx(cipher_on=False, keyed_sites=False,
                            keyed_sign=False),                  ("random", "text"), True),
    ("lsbm_cfa_d4",     _naive_with,                               ("random",),        True),
    ("sim_d4",          _sim_arm,                                ("random",),        False),
]
K8A_PAYLOADS = [0.10, 0.20]

P_K8A_S = os.path.join(OUT_DIR, "K8a_splits.csv")
P_K8A_M = os.path.join(OUT_DIR, "K8a_summary.csv")
P_K8A_W = os.path.join(OUT_DIR, "K8a_wilcoxon.csv")
P_K8A_D = os.path.join(OUT_DIR, "K8a_distortion.csv")
P_K8B_S = os.path.join(OUT_DIR, "K8b_splits.csv")
P_K8B_M = os.path.join(OUT_DIR, "K8b_summary.csv")
P_K8B_D = os.path.join(OUT_DIR, "K8b_distortion.csv")


def _cells_run(cells, keep, covers, develop_fn, path_s, path_d, tag):
    """cells: list of dict(name, fn, kernel_key, kfn, bpc, payload, extra)
    covers: {kernel_key: (covC, covS)} features of the untouched cover."""
    n_pairs = len(keep)
    for cell in cells:
        if out_of_time():
            log("TIME CAP in " + tag); break
        t0 = time.time()
        name, bpc, kk = cell["name"], cell["bpc"], cell["kernel_key"]
        covC, covS = covers[kk]
        sC, sS, ok, drows = [], [], [], []
        for img, raw in keep:
            cd = develop_fn(raw, kk)
            rng = np.random.default_rng(abs(hash((SEED, img, name, bpc,
                                                  cell["payload"], tag))) % (2**32))
            try:
                st, info = cell["fn"](cd, cell["kfn"], bpc, rng)
            except Exception as e:
                st, info = None, dict(status=f"error:{type(e).__name__}")
            if st is None:
                ok.append(False)
                sC.append(np.zeros(covC.shape[1], np.float32))
                sS.append(np.zeros(covS.shape[1], np.float32))
                continue
            a, b = _feat_all(st)
            sC.append(a); sS.append(b); ok.append(True)
            vr = violation_with(st, cell["kfn"]) if cell.get("viol", True) else np.nan
            drows.append(dict(image=img, arm=name, setting=cell["setting"],
                              kernel=kk, bpc=bpc, payload=cell["payload"],
                              psnr=psnr(cd, st),
                              change_rate=info.get("change_rate", np.nan),
                              efficiency=info.get("efficiency", np.nan),
                              ber=info.get("ber", np.nan), viol_rate=vr,
                              extractable=cell["extractable"]))
        ok = np.asarray(ok)
        if ok.mean() < MIN_CELL_FRAC:
            log(f"  {name}@{bpc} {cell['payload']}: {ok.sum()}/{n_pairs} embedded -- refused")
            continue
        pd.DataFrame(drows).to_csv(path_d, mode="a", index=False,
                                   header=not os.path.exists(path_d))
        sC = np.asarray(sC); sS = np.asarray(sS)
        rows = []
        for s in range(RV_SPLITS):
            idx = np.random.default_rng(1000 + s).permutation(n_pairs)
            tr, te = idx[:n_pairs // 2], idx[n_pairs // 2:]
            for det, Xc, Xs in (("cfa", covC, sC), ("spam", covS, sS)):
                trm, tem = tr[ok[tr]], te[ok[te]]
                if len(trm) < 8 or len(tem) < 4:
                    continue
                mu = Xc[trm].mean(0); sd = Xc[trm].std(0) + 1e-12
                ens = FLDEnsemble(L=RV_ENS_L, seed=1000 + s).fit(
                    (Xc[trm] - mu) / sd, (Xs[trm] - mu) / sd,
                    grid=(4, 8, 16, 20) if det == "cfa"
                         else (20, 50, 100, 200, 400))
                cs = ens.score((Xc[tem] - mu) / sd)
                ss = ens.score((Xs[tem] - mu) / sd)
                rows.append(dict(arm=name, setting=cell["setting"], kernel=kk,
                                 bpc=bpc, payload=cell["payload"], detector=det,
                                 split=s, n_pairs=int(ok.sum()),
                                 P_E=round(p_error(cs, ss), 5),
                                 AUC=round(auc(cs, ss), 5)))
        pd.DataFrame(rows).to_csv(path_s, mode="a", index=False,
                                  header=not os.path.exists(path_s))
        log(f"  {tag} {name:<18} bpc={bpc:<5} {cell['payload']:<6} "
            f"{RV_SPLITS} splits ({time.time()-t0:.0f}s)")


def _summarise(path_s, path_m, keys):
    df = pd.read_csv(path_s)
    out = []
    for k, g in df.groupby(keys):
        for col in ("P_E", "AUC"):
            v = g[col].to_numpy(float)
            mu = v.mean(); sd = v.std(ddof=1) if len(v) > 1 else 0.0
            half = 1.96 * sd / np.sqrt(len(v)) if len(v) > 1 else 0.0
            out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)),
                            metric=col, n_splits=len(v), mean=round(mu, 5),
                            sd=round(sd, 5), ci_lo=round(mu - half, 5),
                            ci_hi=round(mu + half, 5)))
    pd.DataFrame(out).to_csv(path_m, index=False)
    return df


def _pairs_and_cover_features(kernel_devs):
    """kernel_devs: {key: callable raw->developed cover}."""
    paths = rv_sample_paths(RV_N_STEG)
    keep, covers_raw = [], {k: ([], []) for k in kernel_devs}
    for i, (path, raw) in enumerate(cover_stream(paths, RV_N_STEG)):
        keep.append((os.path.basename(path), raw))
        for k, dev in kernel_devs.items():
            a, b = _feat_all(dev(raw))
            covers_raw[k][0].append(a); covers_raw[k][1].append(b)
        if (i + 1) % max(1, RV_N_STEG // 10) == 0:
            log(f"    covers {i+1}/{RV_N_STEG}")
        if out_of_time():
            log("TIME CAP in cover features"); break
    covers = {k: (np.asarray(a), np.asarray(b)) for k, (a, b) in covers_raw.items()}
    return keep, covers


def k8a_ablation_steganalysis():
    log("K8a  one-factor ablation + steganalysis (all arms on d4)")
    keep, covers = _pairs_and_cover_features({"d4": lambda raw: develop(raw, "d4")})
    cells = []
    for name, fn, kinds, extractable in K8A_ARMS:
        for bpc in K8A_PAYLOADS:
            for kind in kinds:
                cells.append(dict(name=name, fn=fn, kernel_key="d4",
                                  kfn=DEMOSAIC["d4"], bpc=bpc, payload=kind,
                                  setting=name, extractable=extractable))
    _cells_run(cells, keep, covers, lambda raw, kk: develop(raw, kk),
               P_K8A_S, P_K8A_D, "K8a")
    df = _summarise(P_K8A_S, P_K8A_M, ["arm", "bpc", "payload", "detector"])

    # paired Wilcoxon, split by split: each arm against the full system
    tests = []
    for (b, kind, det), g in df.groupby(["bpc", "payload", "detector"]):
        F = g[g.arm == "full"].sort_values("split")
        if len(F) < 5:
            continue
        for arm in sorted(g.arm.unique()):
            if arm == "full":
                continue
            A = g[g.arm == arm].sort_values("split")
            if len(A) != len(F):
                continue
            a, f = A.P_E.to_numpy(float), F.P_E.to_numpy(float)
            if np.allclose(a, f):
                stat, pv = np.nan, 1.0
            else:
                try:
                    stat, pv = wilcoxon(a, f)
                except Exception:
                    stat, pv = np.nan, np.nan
            tests.append(dict(bpc=b, payload=kind, detector=det, arm=arm,
                              mean_PE_arm=round(a.mean(), 5),
                              mean_PE_full=round(f.mean(), 5),
                              delta=round(a.mean() - f.mean(), 5),
                              wilcoxon_stat=stat, p_value=pv))
    pd.DataFrame(tests).to_csv(P_K8A_W, index=False)
    log(f"K8a done -> {P_K8A_S} {P_K8A_M} {P_K8A_W} {P_K8A_D}")
    return True


def k8b_sensitivity_steganalysis():
    log("K8b  sensitivity + steganalysis")
    KERN = {"support_3x3_bilinear": (demosaic_d1, 3),
            "support_5x5_MHC":      (demosaic_d2, 5),
            "gauss_r4_support_9":   (demosaic_d4, 9),
            "gauss_r6_support_13":  (make_wide_demosaic(6), 13),
            "gauss_r8_support_17":  (make_wide_demosaic(8), 17)}
    devs = {k: (lambda raw, f=f: f(bayer_mosaic(raw))) for k, (f, _) in KERN.items()}
    keep, covers = _pairs_and_cover_features(devs)

    def dev(raw, kk):
        return devs[kk](raw)

    cells = []
    # (a) kernel support, 0.20 bpc, full system
    for kname, (kfn, supp) in KERN.items():
        cells.append(dict(name="kernel", fn=_fx(), kernel_key=kname, kfn=kfn,
                          bpc=0.20, payload="random",
                          setting=f"{kname}", extractable=True))
    # (b) Hamming block exponent, forced, d4, 0.05 bpc (larger p needs carriers)
    d4k = "gauss_r4_support_9"
    for pf in (2, 3, 4, 5):
        cells.append(dict(name="hamming_p", fn=_fx(p_force=pf), kernel_key=d4k,
                          kfn=DEMOSAIC["d4"], bpc=K3_P_SWEEP_BPC, payload="random",
                          setting=f"p={pf}", extractable=True))
    cells.append(dict(name="hamming_p", fn=_fx(), kernel_key=d4k,
                      kfn=DEMOSAIC["d4"], bpc=K3_P_SWEEP_BPC, payload="random",
                      setting="p=auto", extractable=True))
    # (c) carrier policy, d4, 0.20 bpc.  keyed-random is the d4 kernel cell above.
    cells.append(dict(name="carrier_policy", fn=_fx(cost_aware=True),
                      kernel_key=d4k, kfn=DEMOSAIC["d4"], bpc=0.20,
                      payload="random", setting="cost_aware_oracle",
                      extractable=False))
    cells.append(dict(name="carrier_policy", fn=_fx(), kernel_key=d4k,
                      kfn=DEMOSAIC["d4"], bpc=0.20, payload="random",
                      setting="keyed_random", extractable=True))
    _cells_run(cells, keep, covers, dev, P_K8B_S, P_K8B_D, "K8b")
    _summarise(P_K8B_S, P_K8B_M, ["arm", "setting", "bpc", "detector"])
    log(f"K8b done -> {P_K8B_S} {P_K8B_M} {P_K8B_D}")
    return True
