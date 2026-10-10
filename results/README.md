# Result files

Every number in the paper is read from these files by `analysis/mktables.py`
and `analysis/mkfigs.py`. They are released under CC BY 4.0.

| File | Produced by | Used for |
|---|---|---|
| `A_imperceptibility_reseed.csv` | `k7_reseeded_distortion_robustness` | PSNR, SSIM, change rate, efficiency, BER, violation rate (Tables 2–5) |
| `C_robustness_reseed.csv` | `k7_reseeded_distortion_robustness` | bit error rate after each channel operation (robustness table) |
| `K1_*.csv`, `K1b_*.csv` | `k1-k5_revision_cpu` (RUN_K=1), `k1b_payloads_010_040` | P_E and AUC over ten splits, paired Wilcoxon tests |
| `K2_ablation.csv` | `k1-k5_revision_cpu` (RUN_K=2) | first-round ablation, superseded by K8a, kept for the record |
| `K3_sensitivity.csv` | `k1-k5_revision_cpu` (RUN_K=3) | distortion part of the sensitivity table |
| `K4_*.csv` | `k1-k5_revision_cpu` (RUN_K=4) | kernel-mismatch table and figure |
| `K5_cost_profile.csv` | `k1-k5_revision_cpu` (RUN_K=5) | first-round cost profile (timed under memory tracing) |
| `K6c_*.csv` | `k6c_learned_steganalysis_gpu` | learned-detector table and training curves |
| `K8a_*.csv` | `k8a_one_factor_ablation` | one-factor ablation, distortion and detectability |
| `K8b_*.csv` | `k8b_sensitivity_steganalysis` | P_E part of the sensitivity table and figure |
| `K9_*` | `k9_hardware_cost_comparison` | hardware record and cost comparison |

Column meanings: `bpc` is payload in bits per pixel-channel; `P_E` is
min over thresholds of (P_FA + P_MD)/2 on the held-out half of a split;
`viol_rate` is the fraction of interpolated sites (image interior) where
y != D(M(y)); `ber` is the bit error rate of the recovered payload.
