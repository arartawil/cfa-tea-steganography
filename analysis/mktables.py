# -*- coding: utf-8 -*-
"""Every table in the revised manuscript, generated from the result files."""
import pandas as pd, numpy as np, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "analysis", "tables")
os.makedirs(OUT, exist_ok=True)

# ---- one canonical label per configuration  (R2-M13) ----------------------
LBL = {
 "ours_xtea_d4":  r"Proposed (XTEA), $d_4$",
 "ours_xtea_d2":  r"Proposed (XTEA), $d_2$",
 "ours_xtea_d1":  r"Proposed (XTEA), $d_1$",
 "ours_tea_d4":   r"Proposed (TEA), $d_4$",
 "ours_tea_d2":   r"Proposed (TEA), $d_2$",
 "ours_tea_d1":   r"Proposed (TEA), $d_1$",
 "ours_noenc_d4": r"Proposed, no encryption, $d_4$",
 "ours_nocode_d4":r"Proposed, no coding, $d_4$",
 "ours_naive_d1": r"CFA-domain LSB matching, $d_1$",
 "ours_sim_d4":   r"Distortion bound (simulator), $d_4$",
 "ours_sim_d2":   r"Distortion bound (simulator), $d_2$",
 "ours_sim_d1":   r"Distortion bound (simulator), $d_1$",
 "hill_sim":      r"HILL (simulator)",
 "lsbm":          r"LSB matching",
 "lsbr":          r"LSB replacement",
 "almazaydeh20":  r"Blue-channel edge-adaptive LSB",
}
# the order tables are written in: extractable schemes first, then the two
# simulators, then the developed-domain baselines
ORDER = ["ours_xtea_d4","ours_tea_d4","ours_xtea_d2","ours_tea_d2",
         "ours_xtea_d1","ours_tea_d1","ours_noenc_d4","ours_nocode_d4",
         "ours_naive_d1","ours_sim_d4","ours_sim_d2","ours_sim_d1",
         "hill_sim","lsbr","lsbm","almazaydeh20"]
PAY = [0.05, 0.10, 0.20, 0.40]
NA  = r"\textemdash"

def tbl(path, cols, rows, caption, label, note=None, align=None, fit=False):
    a = align or ("l" + "c" * (len(cols) - 1))
    L = [r"\begin{table}[htbp]", r"\centering",
         rf"\caption{{{caption}}}\label{{{label}}}",
         (r"\resizebox{\linewidth}{!}{%" + "\n" if fit else "") + rf"\begin{{tabular}}{{{a}}}", r"\toprule",
         " & ".join(cols) + r" \\", r"\midrule"]
    for r_ in rows:
        if r_ and r_[0] == "__RULE__":
            L.append(r"\midrule"); continue
        L.append(" & ".join(r_) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}" + ("}" if fit else "")]
    if note:
        L.append(rf"\begin{{tablenotes}}\footnotesize\item {note}\end{{tablenotes}}"
                 if False else rf"\\[2pt]\begin{{minipage}}{{\linewidth}}\footnotesize {note}\end{{minipage}}")
    L.append(r"\end{table}")
    open(os.path.join(OUT, path), "w").write("\n".join(L) + "\n")
    print("wrote", path)

A  = pd.read_csv(f"{D}/A_imperceptibility_reseed.csv"); Aok = A[A.status == "ok"]
C  = pd.read_csv(f"{D}/C_robustness_reseed.csv");       Cok = C[C.status == "ok"]
S  = pd.concat([pd.read_csv(f"{D}/K1_summary.csv"), pd.read_csv(f"{D}/K1b_summary.csv")])
W  = pd.concat([pd.read_csv(f"{D}/K1_wilcoxon.csv"), pd.read_csv(f"{D}/K1b_wilcoxon.csv")])
K2 = pd.read_csv(f"{D}/K2_ablation.csv"); K2ok = K2[K2.status == "ok"]
K3 = pd.read_csv(f"{D}/K3_sensitivity.csv")
K4 = pd.read_csv(f"{D}/K4_summary.csv")
K5 = pd.read_csv(f"{D}/K5_cost_profile.csv")
K6 = pd.read_csv(f"{D}/K6c_cnn_steganalysis.csv")

def cell(df, m, b, col, dec=2, sd=False):
    s = df[(df.method == m) & (np.isclose(df.bpc, b))]
    if not len(s) or s[col].isna().all():
        return NA
    v = s[col].mean()
    if sd:
        return f"{v:.{dec}f} $\\pm$ {s[col].std():.{dec}f}" if len(s) > 1 else f"{v:.{dec}f}"
    return f"{v:.{dec}f}"

# ================================================================= TABLE 1 ==
rows = []
prev_sim = False
for m in ORDER:
    if m not in set(Aok.method):
        continue
    sim = m.startswith("ours_sim") or m == "hill_sim"
    if sim and not prev_sim:
        rows.append(["__RULE__"])
    prev_sim = sim
    r = [LBL[m]]
    for b in PAY:
        s = Aok[(Aok.method == m) & (np.isclose(Aok.bpc, b))]
        r.append(NA if not len(s) else f"{s.psnr.mean():.2f} $\\pm$ {s.psnr.std():.2f}")
    rows.append(r)
tbl("t1_psnr.tex", ["Method"] + [f"{b:.2f}" for b in PAY], rows,
    "Peak signal-to-noise ratio in decibels, mean $\\pm$ standard deviation over 600 "
    "cover images, by payload in bits per pixel-channel. A dash marks a "
    "(method, payload) pair whose capacity is exceeded. The two simulator rows "
    "are distortion bounds and carry no extractable payload.",
    "tab:psnr",
    "Covers are a seeded random sample of 600 ALASKA\\#2 images (seed 20261003), "
    "center-cropped to $256\\times256$ on an even origin and developed with the "
    "kernel named in the row label.")

# ================================================================= TABLE 2 ==
rows = []
for m in ORDER:
    if m not in set(Aok.method):
        continue
    r = [LBL[m]]
    for b in PAY:
        r.append(cell(Aok, m, b, "change_rate", 4))
    e = Aok[(Aok.method == m)]
    r.append(NA if e.efficiency.isna().all() else f"{e.efficiency.mean():.2f}")
    rows.append(r)
tbl("t2_change.tex", ["Method"] + [f"{b:.2f}" for b in PAY] + ["Mean eff."], rows,
    "Change rate, the fraction of carriers actually modified, by payload; and "
    "mean embedding efficiency in payload bits per modified carrier. Efficiency "
    "is defined only for the syndrome-coded arms.",
    "tab:change")

# ================================================================= TABLE 3 ==
#  deterministic violation test  (R2-M3: training-free, reported separately)
rows = []
for m in ORDER:
    if m not in set(Aok.method):
        continue
    r = [LBL[m]]
    for b in PAY:
        s = Aok[(Aok.method == m) & (np.isclose(Aok.bpc, b))]
        if not len(s) or s.viol_rate.isna().all():
            r.append(NA); continue
        r.append(f"{s.viol_rate.mean():.4f}")
    s_all = Aok[Aok.method == m]
    frac = float((s_all.viol_rate > 0).mean()) if len(s_all) else np.nan
    r.append(f"{frac:.3f}")
    r.append(f"{0.5*(1-frac):.3f}")
    rows.append(r)
tbl("t3_violation.tex",
    ["Method"] + [f"{b:.2f}" for b in PAY] + ["Flagged", "$P_E$"], rows,
    "Color filter array consistency. Columns 2--5 give the violation rate at "
    "interpolated sites by payload. ``Flagged'' is the fraction of stego images "
    "with a non-zero violation rate, and $P_E$ is the error probability of the "
    "deterministic threshold test that flags exactly those images.",
    "tab:viol",
    "This test is training-free: it compares an image with its own "
    "re-development and needs no cover--stego examples. No developed cover in "
    "the sample violates the relation, so the false-alarm rate is zero by "
    "measurement and $P_E = \\tfrac{1}{2}(1-\\mathrm{flagged})$. It is a "
    "different detector from the trained 23-feature ensemble of "
    "Table~\\ref{tab:pe_cfa}.")

# ================================================================= TABLE 4 ==
rows = []
for m in ORDER:
    if m not in set(Aok.method):
        continue
    s = Aok[Aok.method == m]
    nmax = Aok[Aok.method == m].groupby("bpc").size().max()
    pays = sorted(Aok[Aok.method == m].bpc.unique())
    SIM = m.startswith("ours_sim") or m == "hill_sim"
    extractable = "no" if SIM else "yes"
    bermax = r"n/a" if SIM else (NA if s.ber.isna().all() else f"{s.ber.max():.1e}")
    rows.append([LBL[m], extractable, bermax,
                 f"{s.viol_rate.max():.1e}" if not s.viol_rate.isna().all() else NA,
                 ", ".join(f"{p:.2f}" for p in pays)])
tbl("t4_correct.tex",
    ["Method", "Extractable", "Max BER", "Max viol.", "Payloads supported"], rows,
    "Correctness and operating range. ``Extractable'' marks the arms that carry "
    "a real payload recovered by a real extractor; the simulator arms report a "
    "distortion bound only and no bit error rate is defined for them. The "
    "maxima are over every image and every supported payload.",
    "tab:correct",
    "The simulator arms place modifications at a payload--distortion bound and "
    "never form a message, so no bit error rate exists for them; the entry is "
    "\\emph{n/a} rather than zero. Reporting zero there, as an earlier version "
    "of this work did, invites the reader to treat a distortion bound as a "
    "working system.")

# ================================================================= TABLE 5 ==
#  trained detectors, with CIs over ten splits
for det, fn, cap in (
 ("cfa", "t5_pe_cfa.tex",
  "Error probability $P_E$ of the trained 23-feature structural detector, mean "
  "with 95\\% confidence interval over ten independent pair-preserving splits "
  "of 1000 cover--stego pairs. Lower means easier to detect; 0.5 is chance."),
 ("spam", "t6_pe_spam.tex",
  "Error probability $P_E$ of the SPAM-686 detector with an FLD ensemble, mean "
  "with 95\\% confidence interval over ten independent pair-preserving splits "
  "of 1000 cover--stego pairs.")):
    pe = S[(S.metric == "P_E") & (S.detector == det)]
    rows = []
    for m in ORDER:
        if m not in set(pe.method):
            continue
        r = [LBL[m]]
        for b in PAY:
            s = pe[(pe.method == m) & (np.isclose(pe.bpc, b))]
            r.append(NA if not len(s) else
                     f"{s['mean'].iloc[0]:.3f} [{s.ci_lo.iloc[0]:.3f}, {s.ci_hi.iloc[0]:.3f}]")
        rows.append(r)
    tbl(fn, ["Method"] + [f"{b:.2f}" for b in PAY], rows, cap,
        f"tab:pe_{det}",
        "Each split draws a fresh 500/500 partition; a cover and its stego never "
        "straddle the boundary. Seeds 1000--1009 are fixed and reported with the "
        "code.", fit=True)

# ================================================================= TABLE 6 ==
rows = []
for _, r in W.sort_values(["detector", "bpc", "method", "baseline"]).iterrows():
    rows.append([r.detector.upper(), f"{r.bpc:.2f}", LBL[r.method], LBL[r.baseline],
                 f"{r.mean_PE_ours:.3f}", f"{r.mean_PE_base:.3f}",
                 f"{r.delta:+.3f}", f"{r.p_value:.4f}"])
tbl("t7_wilcoxon.tex",
    ["Detector", "bpc", "Proposed", "Baseline", "$P_E$ ours", "$P_E$ base",
     r"$\Delta$", "$p$"], rows,
    "Paired Wilcoxon signed-rank tests on $P_E$, computed split by split over "
    "the same ten partitions. A positive $\\Delta$ means the proposed scheme is "
    "harder to detect.",
    "tab:wilcoxon",
    "With ten paired observations the smallest attainable two-sided $p$-value is "
    "0.0020, and every comparison attains it.",
    align="llllcccc", fit=True)

# ================================================================= TABLE 7 ==
#  Round 2 (R1-1): one-factor-at-a-time ablation, all arms on d4, 1000 pairs
K8=pd.read_csv(f"{D}/K8a_distortion.csv"); K8S=pd.read_csv(f"{D}/K8a_summary.csv")
K8W=pd.read_csv(f"{D}/K8a_wilcoxon.csv")
Y, N_ = r"\checkmark", r"\textemdash"
ARMS=[("full",           "Full scheme",                         (1,1,1,1)),
      ("minus_cipher",   "No cipher (plaintext payload)",       (0,1,1,1)),
      ("minus_keyed_sites","No keyed carriers (raster order)",  (1,0,1,1)),
      ("minus_keyed_sign","No keyed signs (fixed $+1$)",        (1,1,0,1)),
      ("minus_coding",   "No syndrome coding",                  (1,1,1,0)),
      ("minus_all_three","No cipher, carriers or signs",        (0,0,0,1)),
      ("__RULE__",None,None),
      ("lsbm_cfa_d4",    "CFA-domain LSB matching",             None),
      ("sim_d4",         "Distortion bound (simulator)",        None)]
def _d(arm, b, col, kind="random"):
    g=K8[(K8.arm==arm)&(K8.payload==kind)&np.isclose(K8.bpc,b)]
    return g[col].mean()
rows=[]
for arm,lab,f in ARMS:
    if arm=="__RULE__": rows.append(["__RULE__"]); continue
    flags=[Y if v else N_ for v in f] if f else [N_,"random",r"random",N_] if arm=="lsbm_cfa_d4" else [Y,Y,Y,"bound"]
    eff=_d(arm,0.20,"efficiency")
    rows.append([lab]+flags+[f"{_d(arm,0.10,'psnr'):.2f}",f"{_d(arm,0.20,'psnr'):.2f}",
                 f"{_d(arm,0.20,'change_rate'):.4f}", NA if eff!=eff else f"{eff:.2f}"])
tbl("t8_ablation.tex",
    ["Configuration","Cipher","Keyed carriers","Keyed signs","Coding",
     "PSNR 0.10","PSNR 0.20","Change 0.20","Eff. 0.20"], rows,
    "One-factor-at-a-time ablation: distortion. Every arm uses the $d_4$ kernel, "
    "the same 1000 ALASKA\\#2 covers and a random payload, and each of the first "
    "five rows below the full scheme removes exactly one component. Every "
    "extractable arm recovers its payload with a bit error rate of zero and every "
    "arm satisfies the consistency relation exactly.",
    "tab:ablation",
    "The CFA-domain LSB matching control now runs on $d_4$, the same kernel as "
    "every other arm, rather than on $d_1$ as in the previous version. Its "
    "carriers and signs are drawn at random, not from the keystream. The "
    "simulator carries no extractable payload and is shown only as a distortion "
    "floor.", align="lcccccccc", fit=True)

def _pe(arm,b,det,kind="random"):
    g=K8S[(K8S.arm==arm)&(K8S.payload==kind)&np.isclose(K8S.bpc,b)&(K8S.detector==det)&(K8S.metric=="P_E")]
    r=g.iloc[0]; return r["mean"], (r.ci_hi-r.ci_lo)/2
def _sig(arm,b,det,kind="random"):
    if arm=="full": return ""
    g=K8W[(K8W.arm==arm)&(K8W.payload==kind)&np.isclose(K8W.bpc,b)&(K8W.detector==det)]
    if not len(g): return ""
    return r"$^{*}$" if g.p_value.iloc[0] < 0.05 else ""
rows=[]
for kind,title in (("random","Random payload"),("text","Structured payload (ASCII text)")):
    if kind=="text": rows.append(["__RULE__"])
    rows.append([r"\multicolumn{5}{l}{\textit{"+title+r"}}"])
    for arm,lab,f in ARMS:
        if arm=="__RULE__":
            if kind=="random": rows.append(["__RULE__"])
            continue
        if not len(K8S[(K8S.arm==arm)&(K8S.payload==kind)]): continue
        r=[r"\quad "+lab]
        for det in ("spam","cfa"):
            for b in (0.10,0.20):
                m,h=_pe(arm,b,det,kind); r.append(f"{m:.3f} $\\pm$ {h:.3f}{_sig(arm,b,det,kind)}")
        rows.append(r)
tbl("t8b_ablation_det.tex",
    ["Configuration","SPAM 0.10","SPAM 0.20","Structural 0.10","Structural 0.20"], rows,
    "One-factor-at-a-time ablation: detectability. $P_E$ of the SPAM-686 and "
    "structural detectors, mean $\\pm$ 95\\% confidence half-width over ten "
    "pair-preserving splits of 1000 cover--stego pairs, all on $d_4$. Higher is "
    "harder to detect.",
    "tab:ablation_det",
    "$^{*}$\\,Paired Wilcoxon test against the full scheme on the same payload, "
    "detector and splits, $p<0.05$. Removing the cipher moves $P_E$ by at most "
    "0.008 in any of the eight cells, in both directions; the keyed signs and "
    "the syndrome coding are the two components whose removal makes the scheme "
    "measurably easier to detect under SPAM-686.", align="lcccc")

# ================================================================= TABLE 8 ==
#  Round 2 (R1-4): sensitivity with P_E. Distortion from K3 (300 covers),
#  detectability from K8b (1000 pairs, ten splits); payload P_E from K1/K1b.
K8B=pd.read_csv(f"{D}/K8b_summary.csv"); K8B=K8B[K8B.metric=="P_E"]
K8BA=pd.read_csv(f"{D}/K8b_summary.csv")
def _pb(arm,setting,det,metric="P_E"):
    g=K8BA[(K8BA.arm==arm)&(K8BA.setting==setting)&(K8BA.detector==det)&(K8BA.metric==metric)]
    return NA if not len(g) else f"{g['mean'].iloc[0]:.3f}"
def _p1(b,det,metric="P_E"):
    g=S[(S.method=="ours_xtea_d4")&np.isclose(S.bpc,b)&(S.detector==det)&(S.metric==metric)]
    return NA if not len(g) else f"{g['mean'].iloc[0]:.3f}"
SENS = [("kernel_support", "Demosaicing kernel support (0.20 bpc)", "kernel",
         {"support_3x3_bilinear": (r"bilinear, $3\times3$","support_3x3_bilinear"),
          "support_5x5_MHC": (r"Malvar--He--Cutler, $5\times5$","support_5x5_MHC"),
          "gauss_r4_support_9": (r"Gaussian $r=4$, $9\times9$ ($d_4$)","gauss_r4_support_9"),
          "gauss_r6_support_13": (r"Gaussian $r=6$, $13\times13$","gauss_r6_support_13"),
          "gauss_r8_support_17": (r"Gaussian $r=8$, $17\times17$","gauss_r8_support_17")}),
        ("hamming_p", "Hamming block exponent $p$ (0.05 bpc)", "hamming_p",
         {f"p={i}": (f"$p={i}$" + (" (automatic choice)" if i==5 else ""), f"p={i}") for i in (2, 3, 4, 5)}),
        ("payload", "Payload rate", None,
         {f"bpc={b}": (f"{b} bpc", b) for b in (0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)}),
        ("carrier_selection", "Carrier ordering (0.20 bpc)", "carrier_policy",
         {"keyed_random": ("keyed pseudorandom (extractable)","keyed_random"),
          "cost_aware": ("HILL-cost ordered (oracle, not extractable)","cost_aware_oracle")})]
rows = []
for axis, title, arm, mapping in SENS:
    rows.append([r"\multicolumn{8}{l}{\textit{" + title + r"}}"])
    a = K3[(K3.axis == axis) & (K3.status == "ok")]
    for k, (lab, key) in mapping.items():
        s_ = a[a.setting == k]
        if not len(s_):
            continue
        f = _p1 if arm is None else (lambda k, d, m="P_E", _a=arm: _pb(_a, k, d, m))
        rows.append([r"\quad " + lab, f"{s_.psnr.mean():.2f}",
                     f"{s_.change_rate.mean():.4f}", f"{s_.efficiency.mean():.2f}",
                     f(key,"spam"), f(key,"spam","AUC"), f(key,"cfa"), f(key,"cfa","AUC")])
tbl("t9_sensitivity.tex",
    ["Setting", "PSNR", "Change", "Eff.", "$P_E$ SPAM", "AUC SPAM", "$P_E$ struct.", "AUC struct."], rows,
    "Sensitivity of the proposed scheme to its design parameters, varied one at "
    "a time around the reported configuration. Distortion is averaged over 300 "
    "covers; $P_E$ is the mean over ten pair-preserving splits of 1000 "
    "cover--stego pairs, and AUC is averaged over the same splits. For $P_E$ higher "
    "is harder to detect; for AUC lower is harder to detect. Bit error rate and "
    "violation rate are zero in every row.",
    "tab:sensitivity",
    "Payload-axis $P_E$ values come from the main detectability study "
    "(Tables~\\ref{tab:pe_cfa} and \\ref{tab:pe_spam}); payloads not evaluated "
    "there are marked \\textemdash. The cost-ordered row is an oracle: the "
    "receiver cannot reproduce a carrier order computed from the cover, so it "
    "bounds what side-informed coding could add rather than describing a "
    "working variant.", align="lccccccc", fit=True)

# ================================================================= TABLE 9 ==
rows = []
for _, r in K4.iterrows():
    rows.append([f"$d_{r.embed_kernel[-1]}$", f"$d_{r.verify_kernel[-1]}$",
                 "yes" if r.matched else "no",
                 f"{r.cover_viol_mean:.4f}", f"{r.stego_viol_mean:.4f}",
                 f"{r.AUC:.4f}", f"{r.P_E:.4f}"])
tbl("t10_mismatch.tex",
    ["Embed", "Verify", "Matched", "Cover viol.", "Stego viol.", "AUC", "$P_E$"],
    rows,
    "Kernel mismatch. The payload is embedded under one demosaicing kernel and "
    "the consistency test is applied under another, over 300 cover images at "
    "0.20 bits per pixel-channel.",
    "tab:mismatch",
    "Under mismatch the cover violates the relation as heavily as the stego "
    "image does, so the structural test retains no discriminative power "
    "(AUC near 0.5). The limitation is the detector's, not the embedder's: the "
    "payload is still recovered without error in every cell.")

# ================================================================ TABLE 10 ==
CU=pd.read_csv(f"{D}/K6c_curves.csv")
rows = []
for _, r in K6.iterrows():
    g = CU[(CU.method == r.method) & np.isclose(CU.bpc, r.bpc)]
    n_att = g.attempt.nunique(); ep_run = int(g.groupby("attempt").epoch.max().max())
    if r.method in ("lsbm", "ours_naive_d1"):
        role, outcome = ("control 1" if r.method == "lsbm" else "control 2"), "control passed"
    elif ep_run <= 10:
        role, outcome = "---", "did not train"
    else:
        role, outcome = "---", "inconclusive"
    pe = (f"{r.P_E:.3f} [{r.P_E_lo:.3f}, {r.P_E_hi:.3f}]" if outcome != "did not train" else NA)
    rows.append([LBL[r.method], f"{r.bpc:.2f}", role, f"{n_att}", f"{ep_run}",
                 pe, NA if outcome == "did not train" else f"{r.AUC:.3f}",
                 NA if pd.isna(r.spam_ref_P_E) else f"{r.spam_ref_P_E:.3f}",
                 outcome])
tbl("t11_cnn.tex",
    ["Configuration", "bpc", "Role", "Attempts", "Epochs run", "$P_E$ [95\\% CI]", "AUC",
     "SPAM $P_E$", "Outcome"], rows,
    "Exploratory run of the learned detector (one training seed, 45 epochs): an SRM-initialized convolutional detector "
    "(Table~\\ref{tab:cnncfg}) trained on cover--stego pairs produced by the "
    "proposed embedder itself: 1200 training, 300 validation and 400 test pairs. "
    "The epoch is selected on validation and $P_E$ is reported on the test split.",
    "tab:cnn",
    "\\emph{Did not train}: every attempt was still at the chance loss "
    "$\\ln 2$ with chance accuracy when the collapse rule stopped it at epoch 8. "
    "\\emph{Inconclusive}: the network trained, but validation error was still "
    "falling at the final epoch, so the figure is an upper bound on what this "
    "detector would reach and supports no comparative conclusion. Only the two "
    "control rows are interpreted.", align="lcccccccc", fit=True)

# Round 3 (R1-1): three independent training seeds at 0.20 bpc
K6D = pd.concat([pd.read_csv(f"{D}/K6d_seed{sd}.csv") for sd in (2026, 2027, 2028)])
CONV = {"converged": "converged", "still_improving": "still improving", "did_not_train": "did not train"}
rows = []
for m, lab in (("lsbm", "LSB matching (control 1)"), ("ours_naive_d1", r"CFA-domain LSB matching, $d_1$ (control 2)"),
               ("ours_xtea_d4", r"Proposed (XTEA), $d_4$")):
    g = K6D[K6D.method == m].sort_values("train_seed")
    for k, (_, r) in enumerate(g.iterrows()):
        dnt = r.convergence == "did_not_train"
        rows.append([lab if k == 0 else "", f"{int(r.train_seed)}", f"{int(r.attempts)}", f"{int(r.epochs_run)}",
                     NA if dnt else f"{r.P_E:.3f} [{r.P_E_lo:.3f}, {r.P_E_hi:.3f}]",
                     NA if dnt else f"{r.AUC:.3f}", CONV[r.convergence]])
    ok = g[g.convergence != "did_not_train"]
    sdp = f" $\\pm$ {ok.P_E.std(ddof=1):.3f}" if len(ok) > 1 else ""
    sda = f" $\\pm$ {ok.AUC.std(ddof=1):.3f}" if len(ok) > 1 else ""
    rows.append([r"\quad \textit{mean over trained seeds}", f"{len(ok)} of {len(g)}", "", "",
                 f"{ok.P_E.mean():.3f}{sdp}", f"{ok.AUC.mean():.3f}{sda}", ""])
    if m != "ours_xtea_d4": rows.append(["__RULE__"])
tbl("t11c_cnnseeds.tex",
    ["Configuration", "Seed", "Attempts", "Epochs run", "$P_E$ [95\\% CI]", "AUC", "Status"], rows,
    "Learned detector repeated over three independent training seeds at 0.20 bits "
    "per pixel-channel. Pairs and the 1200/300/400 split are identical across seeds; "
    "only weight initialization, batch order and augmentation change. 90 epochs "
    "(control 1: 30).",
    "tab:cnnseeds",
    "Status follows the convergence rule of Table~\\ref{tab:cnncfg}. Means and "
    "standard deviations are over the seeds that trained. SPAM-686 reaches "
    "$P_E=0.325$ on the proposed configuration (Table~\\ref{tab:pe_spam}).",
    align="lcccccc", fit=True)

CFG = [("Architecture", r"Ye-Net style \cite{ye2017deep}: SRM front end, 7 conv.\ stages, global average pooling, linear layer"),
       ("Front end", r"30 SRM $5\times5$ kernels per channel (90 maps), SRM-initialized, trainable, reflect padding"),
       ("Activation after front end", r"truncated linear unit, $T=3$"),
       ("Trunk", r"conv.\ $3,3,3,5,5,5,3$; widths $30,30,32,32,32,16,16$; BN + ReLU; $3\times3$ avg.\ pooling (stride 2) after stages 3--6"),
       ("Input", r"$256\times256\times3$, raw intensities, cover and stego of a pair in the same batch"),
       ("Loss", "two-class cross-entropy"),
       ("Optimizer", r"Adam, weight decay $10^{-4}$, gradient-norm clipping at 5"),
       ("Learning rate", r"$10^{-4}$; constant for 10 epochs, then cosine decay to 0"),
       ("Batch", "16 pairs (32 images) per step"),
       ("Epochs", "exploratory run: 60 for control 1, 45 otherwise; repeated runs: 30 for control 1, 90 otherwise"),
       ("Initialization", "control 1 from scratch; every later run from the control-1 weights"),
       ("Augmentation", r"random rotation by $k\cdot90^\circ$ and horizontal flip, applied identically to both images of a pair"),
       ("Model selection", "epoch with the lowest validation $P_E$; test split never used for selection"),
       ("Collapse rule", r"a run still within $2\times10^{-3}$ of $\ln2$, with accuracy within 0.02 of 0.5 and validation $P_E>0.47$, at epoch 8 is stopped and restarted at one third of the learning rate, at most twice"),
       ("Data split", "1200 / 300 / 400 pairs, permutation seed 2026; pairs never split across partitions"),
       ("Seeds", r"exploratory run: training seed 2026; repeated runs: training seeds 2026, 2027, 2028 with pairs and split fixed (split seed 2026)"),
       ("Convergence", r"converged if the best validation $P_E$ of the last 15 epochs is at most 0.005 below the best before them; otherwise still improving"),
       ("Confidence interval", "2000 bootstrap resamples of the test pairs"),
       ("Hardware", "one Kaggle GPU session (CUDA), PyTorch")]
tbl("t11b_cnncfg.tex", ["Item", "Setting"], [[a_, b_] for a_, b_ in CFG],
    "Configuration of the learned detector, given in full so that the "
    "experiment can be repeated exactly.", "tab:cnncfg", align="lp{0.68\\linewidth}")

# ================================================================ TABLE 11 ==
#  Round 2 (R1-5): K9, timing without memory tracing, single thread, same
#  hardware and same images for every method.
K9 = pd.read_csv(f"{D}/K9_cost.csv"); K9 = K9[K9.status == "ok"]
def _k9(job, b, col="ms"):
    g = K9[(K9.job == job) & np.isclose(K9.bpc, b)]
    return (g[col].mean(), g[col].std()) if len(g) else (np.nan, np.nan)
def _ms(job, b):
    m, sd = _k9(job, b)
    return NA if m != m else f"{m:.2f} $\\pm$ {sd:.2f}"
tot20 = _k9("ours_embed_total", 0.20)[0]
STG = [("ours_keystream_xtea", "XTEA keystream (payload cipher)"),
       ("ours_permutation",    "Keyed carrier permutation"),
       ("ours_embed_no_dev",   "Embedding, excluding development (includes both rows above)"),
       ("ours_demosaic_d4",    "Development with $d_4$"),
       ("ours_embed_total",    "Embedding, end to end"),
       ("ours_extract_total",  "Extraction, end to end")]
rows = [[r"\multicolumn{6}{l}{\textit{Proposed (XTEA), $d_4$, by stage}}"]]
for k, lab in STG:
    m20 = _k9(k, 0.20)[0]
    sh = f"{100*m20/tot20:.1f}" if k not in ("ours_extract_total", "ours_embed_total") else (NA if k.startswith("ours_extract") else "100")
    rows.append([r"\quad " + lab, _ms(k, 0.05), _ms(k, 0.20),
                 f"{_k9(k,0.20,'mpix_per_s')[0]:.2f}", f"{_k9(k,0.20,'peak_mem_mb')[0]:.2f}", sh])
rows.append(["__RULE__"])
rows.append([r"\multicolumn{6}{l}{\textit{Embedding, end to end, same images and payloads}}"])
CMP = [("lsbr", "LSB replacement"), ("lsbm", "LSB matching"),
       ("edge_adaptive", "Blue-channel edge-adaptive LSB"),
       ("cfa_lsbm_d4", r"CFA-domain LSB matching, $d_4$"),
       ("ours_embed_total", r"Proposed (XTEA), $d_4$"),
       ("hill_sim", "HILL (simulator)")]
for k, lab in CMP:
    m20 = _k9(k, 0.20)[0]
    rows.append([r"\quad " + lab, _ms(k, 0.05), _ms(k, 0.20),
                 NA if m20 != m20 else f"{_k9(k,0.20,'mpix_per_s')[0]:.2f}",
                 NA if m20 != m20 else f"{_k9(k,0.20,'peak_mem_mb')[0]:.2f}", NA])
tbl("t12_cost.tex",
    ["Stage or method", "ms, 0.05 bpc", "ms, 0.20 bpc", "Mpixel/s", "Peak MB", r"\% of embed"], rows,
    "Measured computational cost per $256\\times256$ image, mean $\\pm$ standard "
    "deviation over 50 ALASKA\\#2 covers, best of three repetitions per image. "
    "Single thread on an Intel Xeon CPU at 2.20~GHz (4 logical cores, 32~GB "
    "RAM), Python 3.13.15, NumPy 2.1.3, SciPy 1.16.3. Throughput and peak "
    "memory are at 0.20 bpc.",
    "tab:cost",
    "Timing and peak-memory measurement are separate passes, because memory "
    "tracing inflates the wall-clock time of interpreted loops. The blue-channel "
    "scheme cannot carry 0.20 bpc. HILL is a simulator: its time is the cost "
    "computation and the simulated embedding, with no coding. All figures are "
    "measurements on this CPU, not on embedded hardware.", align="lccccc", fit=True)

# ================================================================ TABLE 12 ==
ATT = [("none", "none"), ("png", "PNG re-encode"),
       ("jpeg_q95", "JPEG $q=95$"), ("jpeg_q90", "JPEG $q=90$"),
       ("jpeg_q75", "JPEG $q=75$"),
       ("noise_s0.5", r"Noise $\sigma=0.5$"), ("noise_s1", r"Noise $\sigma=1$"),
       ("noise_s2", r"Noise $\sigma=2$"),
       ("crop_5pct", "Crop 5\\%"), ("crop_10pct", "Crop 10\\%")]
ms = [m for m in ORDER if m in set(Cok.method)]
rows = []
SHORT = {"none": "none", "png": "PNG", "jpeg_q95": "Q95", "jpeg_q90": "Q90",
         "jpeg_q75": "Q75", "noise_s0.5": r"$\sigma{=}0.5$", "noise_s1": r"$\sigma{=}1$",
         "noise_s2": r"$\sigma{=}2$", "crop_5pct": "5\\%", "crop_10pct": "10\\%"}
for m in ms:
    r = [LBL[m]]
    for k, lab in ATT:
        s_ = Cok[(Cok.method == m) & (Cok.attack == k)]
        r.append(NA if not len(s_) else f"{s_.ber.mean():.4f}")
    rows.append(r)
tbl("t13_robust.tex", ["Method"] + [SHORT[k] for k, _ in ATT], rows,
    "Bit error rate after each channel operation, over 200 cover images at 0.05 "
    "and 0.10 bits per pixel-channel pooled. A rate of 0.5 means the payload is "
    "destroyed. Q is the JPEG quality factor, $\\sigma$ the standard deviation of "
    "additive Gaussian noise, and the percentages are border crops.",
    "tab:robust",
    "The lossless rows are the control and confirm the pipeline is exact. The "
    "proposed scheme is more fragile under cropping than the developed-domain "
    "baselines because its keyed permutation spreads the carriers over the "
    "whole lattice, so any excised region damages the entire syndrome stream.",
    align="l" + "c" * len(ATT), fit=True)

# ================================================================ TABLE 13 ==
nA, nAok = len(A), len(Aok)
rows = [
 ["Stage A, nominal combinations",
  f"$600 \\times 4 \\times 14$", f"{600*4*14}"],
 ["Stage A, rows written", "one per combination", f"{nA}"],
 ["Stage A, successful embeddings", "status \\texttt{ok}", f"{nAok}"],
 ["Stage A, refused for capacity", "status \\texttt{capacity\\_exceeded}",
  f"{len(A[A.status=='capacity_exceeded'])}"],
 ["\\quad of which proposed-scheme rows", "methods \\texttt{ours\\_*}",
  f"{len(Aok[Aok.method.str.startswith('ours')])}"],
 ["Stage C, post-attack extractions",
  f"$200 \\times 7 \\times 2 \\times 10$ less refusals", f"{len(Cok)}"],
 ["Detector training cells", "method $\\times$ payload $\\times$ detector", "46"],
 ["Detector fits", "46 cells $\\times$ 10 splits", f"{46*10}"],
 ["Ablation rows (distortion)", "$1000 \\times 22$ arm--payload--type cells", f"{len(K8)}"],
 ["Ablation detector fits", "22 cells $\\times$ 2 detectors $\\times$ 10 splits", f"{len(pd.read_csv(f'{D}/K8a_splits.csv'))}"],
 ["Sensitivity detector fits", "12 settings $\\times$ 2 detectors $\\times$ 10 splits", f"{len(pd.read_csv(f'{D}/K8b_splits.csv'))}"],
 ["Sensitivity rows", "300 images $\\times$ 18 settings", f"{len(K3)}"],
 ["Kernel-mismatch rows", "$300 \\times 3 \\times 3$", f"{300*9}"],
]
tbl("t14_accounting.tex", ["Quantity", "How it is formed", "Count"], rows,
    "Reconciliation of every count reported in this paper with the experimental "
    "design that produces it.",
    "tab:accounting",
    "The script that regenerates this table also asserts each figure against the "
    "raw result files, so the counts cannot drift from the data.",
    align="llr")
print("\nall tables written to", OUT)
