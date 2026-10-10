# =============================================================================
#  K9  --  ROUND 2, Reviewer 1 comment 5: hardware, environment, and a
#  cost comparison with conventional embedders on the SAME hardware and at
#  the SAME payloads.  Timing is measured WITHOUT memory tracing (tracing
#  inflates wall-clock); peak memory is measured in a separate pass.
# =============================================================================
import platform, tracemalloc, json, subprocess

P_K9  = os.path.join(OUT_DIR, "K9_cost.csv")
P_K9E = os.path.join(OUT_DIR, "K9_environment.json")
K9_N_IMG = 50
K9_PAY = [0.05, 0.20]


def k9_environment():
    env = dict(platform=platform.platform(), machine=platform.machine(),
               python=platform.python_version(), os_cpu_count=os.cpu_count(),
               threads_env={k: os.environ.get(k) for k in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                             "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    try:
        txt = open("/proc/cpuinfo").read()
        names = [l.split(":", 1)[1].strip() for l in txt.splitlines()
                 if l.startswith("model name")]
        mhz = [l.split(":", 1)[1].strip() for l in txt.splitlines()
               if l.startswith("cpu MHz")]
        env.update(cpu_model=names[0] if names else None,
                   logical_cpus=len(names), cpu_mhz=mhz[0] if mhz else None)
        flags = [l for l in txt.splitlines() if l.startswith("flags")]
        if flags:
            f = flags[0]
            env["simd"] = [x for x in ("sse4_2", "avx", "avx2", "avx512f")
                           if f" {x}" in f]
    except Exception as e:
        env["cpuinfo_error"] = str(e)
    try:
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")]
        env["mem_total"] = mem[0].split(":", 1)[1].strip() if mem else None
    except Exception:
        pass
    import numpy, scipy, PIL, skimage
    env["libs"] = dict(numpy=numpy.__version__, scipy=scipy.__version__,
                       pillow=PIL.__version__, scikit_image=skimage.__version__,
                       pandas=pd.__version__)
    try:
        env["numpy_blas"] = str(numpy.show_config(mode="dicts")
                                .get("Build Dependencies", {}).get("blas", {}))
    except Exception:
        pass
    json.dump(env, open(P_K9E, "w"), indent=2)
    print(json.dumps(env, indent=2))
    return env


def _time_best(fn, reps=3):
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1000.0


def _peak_mb(fn):
    tracemalloc.start(); fn(); peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 1e6


def k9_cost_comparison():
    log("K9  cost comparison on identical hardware")
    k9_environment()
    paths = rv_sample_paths(K9_N_IMG)
    covers = [c for _, c in cover_stream(paths, K9_N_IMG)]
    ident = lambda m: m
    rows = []
    for i, raw in enumerate(covers):
        cd4 = develop(raw, "d4")
        mp = cd4.shape[0] * cd4.shape[1] / 1e6
        for bpc in K9_PAY:
            nb = int(round(bpc * cd4.size)) // 8
            msg = bytes(np.random.default_rng(i).integers(0, 256, nb, dtype=np.uint8))
            nc = cd4.shape[0] * cd4.shape[1]
            try:
                st, _ = cfa_embed(cd4, msg, XTEA_KEY, DEMOSAIC["d4"], 7, "xtea")
            except Exception:
                st = None
            jobs = {
                # ---- proposed, decomposed ----
                "ours_keystream_xtea": lambda: encrypt_payload(msg, XTEA_KEY, 7, "xtea"),
                "ours_permutation":    lambda: keyed_permutation(nc, XTEA_KEY, 7, "xtea"),
                "ours_embed_no_dev":   lambda: cfa_embed(cd4, msg, XTEA_KEY, ident, 7, "xtea"),
                "ours_demosaic_d4":    lambda: demosaic_d4(bayer_mosaic(cd4).astype(float)),
                "ours_embed_total":    lambda: cfa_embed(cd4, msg, XTEA_KEY, DEMOSAIC["d4"], 7, "xtea"),
                "ours_extract_total":  (lambda: cfa_extract(st, nb, XTEA_KEY, 7, "xtea")) if st is not None else None,
                # ---- conventional embedders, same image, same payload ----
                "lsbr":          lambda: lsbr(cd4, "d4", bpc, np.random.default_rng(i)),
                "lsbm":          lambda: lsbm(cd4, "d4", bpc, np.random.default_rng(i)),
                "edge_adaptive": lambda: almazaydeh20(cd4, "d4", bpc, np.random.default_rng(i)),
                "hill_sim":      lambda: hill_sim(cd4, "d4", bpc, np.random.default_rng(i)),
                "cfa_lsbm_d4":   lambda: _naive_with(cd4, DEMOSAIC["d4"], bpc, np.random.default_rng(i)),
            }
            for name, fn in jobs.items():
                if fn is None:
                    continue
                try:
                    out = fn()
                    status = "ok"
                    if isinstance(out, tuple) and len(out) == 2 and out[0] is None:
                        status = (out[1] or {}).get("status", "failed")
                    ms = _time_best(fn) if status == "ok" else np.nan
                    pk = _peak_mb(fn) if status == "ok" else np.nan
                except Exception as e:
                    status, ms, pk = f"error:{type(e).__name__}", np.nan, np.nan
                rows.append(dict(image=i, bpc=bpc, job=name, status=status,
                                 payload_bytes=nb, megapixels=round(mp, 4),
                                 ms=ms, mpix_per_s=(mp / (ms / 1000.0)) if ms == ms else np.nan,
                                 peak_mem_mb=pk))
        if (i + 1) % 10 == 0:
            log(f"  K9 {i+1}/{len(covers)}")
    df = pd.DataFrame(rows)
    df.to_csv(P_K9, index=False)
    print(df[df.status == "ok"].groupby(["bpc", "job"])[["ms", "mpix_per_s", "peak_mem_mb"]]
          .agg(["mean", "std"]).round(3).to_string())
    print(df.groupby(["bpc", "job"]).status.value_counts().to_string())
    log(f"K9 done -> {P_K9}  {P_K9E}")
    return True
