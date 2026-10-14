# -*- coding: utf-8 -*-
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np, os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D=os.path.join(ROOT,"results"); F=os.path.join(ROOT,"analysis","figures"); os.makedirs(F, exist_ok=True)

# Okabe-Ito: colour-blind safe, eight hues, assigned in fixed order
CB = ["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#56B4E9","#000000","#F0E442"]
plt.rcParams.update({
 "font.size":13, "axes.labelsize":14, "axes.titlesize":14,
 "xtick.labelsize":12, "ytick.labelsize":12, "legend.fontsize":11.5,
 "axes.linewidth":0.9, "lines.linewidth":2.0, "lines.markersize":7,
 "axes.grid":True, "grid.alpha":0.25, "grid.linewidth":0.7,
 "axes.spines.top":False, "axes.spines.right":False,
 "figure.dpi":200, "savefig.bbox":"tight", "pdf.fonttype":42,
 "font.family":"serif"})
def save(fig, stem):
    fig.savefig(f"{F}/{stem}.pdf"); plt.close(fig); print("wrote", stem)

LBL={"ours_xtea_d4":"Proposed (XTEA) $d_4$","ours_tea_d4":"Proposed (TEA) $d_4$",
     "ours_sim_d4":"Distortion bound (sim.)","hill_sim":"HILL (sim.)",
     "lsbm":"LSB matching","lsbr":"LSB replacement",
     "almazaydeh20":"Blue-channel LSB","ours_naive_d1":"CFA-domain LSBM",
     "ours_noenc_d4":"No encryption","ours_nocode_d4":"No coding"}
MK=dict(zip(LBL,["o","s","^","D","v","P","X","*","h","<"]))
CO=dict(zip(LBL,CB+CB))

# ---------------------------------------------------------------- FIG 1 ----
S=pd.concat([pd.read_csv(f"{D}/K1_summary.csv"),pd.read_csv(f"{D}/K1b_summary.csv")])
pe=S[S.metric=="P_E"]
show=["ours_xtea_d4","ours_tea_d4","ours_sim_d4","hill_sim","lsbm","almazaydeh20"]
fig,axes=plt.subplots(1,2,figsize=(11.2,4.3),sharey=True)
for ax,det,t in zip(axes,["cfa","spam"],
                    ["Structural detector (23 features)","SPAM-686"]):
    d=pe[pe.detector==det]
    for m in show:
        g=d[d.method==m].sort_values("bpc")
        if not len(g): continue
        ax.errorbar(g.bpc,g["mean"],
                    yerr=[g["mean"]-g.ci_lo,g.ci_hi-g["mean"]],
                    marker=MK[m],color=CO[m],capsize=3.5,capthick=1.2,
                    label=LBL[m],elinewidth=1.2)
    ax.axhline(0.5,ls=":",c="0.45",lw=1.3)
    ax.set_xlabel("Payload (bits per pixel-channel)"); ax.set_title(t)
    ax.set_xticks([0.05,0.10,0.20,0.40]); ax.set_ylim(-0.03,0.55)
axes[0].set_ylabel("$P_E$   (0.5 = chance)")
axes[0].text(0.055,0.515,"chance",fontsize=10.5,color="0.35")
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(0.5,-0.14))
save(fig,"F1_pe_vs_payload")

# ---------------------------------------------------------------- FIG 2 ----
A=pd.read_csv(f"{D}/A_imperceptibility_reseed.csv"); A=A[A.status=="ok"]
fig,ax=plt.subplots(figsize=(7.4,4.4))
for m in ["lsbm","lsbr","almazaydeh20","hill_sim"]:
    g=A[A.method==m].groupby("bpc").viol_rate.mean()
    if not len(g): continue
    ax.plot(g.index,g.values,marker=MK[m],color=CO[m],label=LBL[m])
ax.plot([0.05,0.10,0.20,0.40],[1e-5]*4,marker="o",color=CO["ours_xtea_d4"],
        label="All proposed arms (exactly 0)",lw=2.6)
ax.set_yscale("log"); ax.set_ylim(5e-6,1.0)
ax.set_yticks([1e-5,1e-4,1e-3,1e-2,1e-1,1])
ax.set_yticklabels(["0","$10^{-4}$","$10^{-3}$","$10^{-2}$","$10^{-1}$","$1$"])
ax.set_xticks([0.05,0.10,0.20,0.40])
ax.set_xlabel("Payload (bits per pixel-channel)")
ax.set_ylabel("Violation rate at interpolated sites")
ax.legend(frameon=False,loc="center left",bbox_to_anchor=(0.02,0.42))
save(fig,"F2_violation")

# ---------------------------------------------------------------- FIG 3 ----
#  Round 2: one-factor ablation, every arm on d4, distortion AND detectability
K8=pd.read_csv(f"{D}/K8a_distortion.csv"); K8s=pd.read_csv(f"{D}/K8a_summary.csv")
arms=["full","minus_cipher","minus_keyed_sites","minus_keyed_sign","minus_coding",
      "minus_all_three","lsbm_cfa_d4","sim_d4"]
names=["Full","No\ncipher","No keyed\ncarriers","No keyed\nsigns","No\ncoding",
       "None of\nthe three","CFA\nLSBM $d_4$","Bound\n(sim.)"]
col=[CB[0]]+[CB[5]]*5+[CB[3],CB[6]]
x=np.arange(len(arms))
fig,axes=plt.subplots(1,2,figsize=(15.5,4.8))
r=K8[(K8.payload=="random")&np.isclose(K8.bpc,0.20)]
ps=[r[r.arm==a].psnr.mean() for a in arms]
axes[0].bar(x,ps,color=col,width=0.64,edgecolor="white",linewidth=2)
for xi,v in zip(x,ps): axes[0].text(xi,v+0.06,f"{v:.2f}",ha="center",fontsize=10.5)
axes[0].set_ylim(53.5,59.6); axes[0].set_ylabel("PSNR (dB)")
axes[0].set_xticks(x); axes[0].set_xticklabels(names,fontsize=9.6)
axes[0].set_title("Distortion, 0.20 bpc, random payload")
q=K8s[(K8s.metric=="P_E")&(K8s.detector=="spam")&(K8s.payload=="random")&np.isclose(K8s.bpc,0.20)].set_index("arm")
m=[q.loc[a,"mean"] for a in arms]
e=[[q.loc[a,"mean"]-q.loc[a,"ci_lo"] for a in arms],[q.loc[a,"ci_hi"]-q.loc[a,"mean"] for a in arms]]
axes[1].bar(x,m,color=col,width=0.64,edgecolor="white",linewidth=2,yerr=e,capsize=4,
            error_kw=dict(lw=1.2,capthick=1.2))
for xi,v in zip(x,m): axes[1].text(xi,v+0.008,f"{v:.3f}",ha="center",fontsize=10.5)
axes[1].axhline(q.loc["full","mean"],color=CB[0],lw=1,ls="--")
axes[1].set_ylim(0.25,0.43); axes[1].set_ylabel("$P_E$ under SPAM-686")
axes[1].set_xticks(x); axes[1].set_xticklabels(names,fontsize=9.6)
axes[1].set_title("Detectability, 0.20 bpc (95% CI, 10 splits)")
save(fig,"F3_ablation")

# ---------------------------------------------------------------- FIG 4 ----
#  Round 2: sensitivity with detectability; no twin axes
K3=pd.read_csv(f"{D}/K3_sensitivity.csv")
B=pd.read_csv(f"{D}/K8b_summary.csv"); B=B[B.metric=="P_E"]
KS=[("support_3x3_bilinear",3),("support_5x5_MHC",5),("gauss_r4_support_9",9),
    ("gauss_r6_support_13",13),("gauss_r8_support_17",17)]
fig,axes=plt.subplots(1,3,figsize=(16.0,4.5))
kp=[K3[(K3.axis=="kernel_support")&(K3.setting==k)].psnr.mean() for k,_ in KS]
lab=["3$\\times$3\nbil.","5$\\times$5\nMHC","9$\\times$9\n$r$=4","13$\\times$13\n$r$=6","17$\\times$17\n$r$=8"]
x=np.arange(5)
axes[0].plot(x,kp,marker="o",color=CB[0])
for xi,v in zip(x,kp): axes[0].annotate(f"{v:.2f}",(xi,v),textcoords="offset points",xytext=(0,9),ha="center",fontsize=10.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(lab,fontsize=10.5); axes[0].set_ylabel("PSNR (dB)")
axes[0].set_ylim(52.5,58.8); axes[0].set_title("Kernel: distortion, 0.20 bpc")
for det,c,mk,nm in (("spam",CB[1],"s","SPAM-686"),("cfa",CB[2],"^","Structural")):
    g=B[(B.arm=="kernel")&(B.detector==det)].set_index("setting")
    mm=[g.loc[k,"mean"] for k,_ in KS]
    axes[1].errorbar(x,mm,yerr=[[g.loc[k,"mean"]-g.loc[k,"ci_lo"] for k,_ in KS],[g.loc[k,"ci_hi"]-g.loc[k,"mean"] for k,_ in KS]],
                     marker=mk,color=c,capsize=3.5,label=nm)
axes[1].set_xticks(x); axes[1].set_xticklabels(lab,fontsize=10.5); axes[1].set_ylabel("$P_E$")
axes[1].set_ylim(0.25,0.45); axes[1].legend(frameon=False,loc="lower right"); axes[1].set_title("Kernel: detectability, 0.20 bpc")
ps=[2,3,4,5]
for det,c,mk,nm in (("spam",CB[1],"s","SPAM-686"),("cfa",CB[2],"^","Structural")):
    g=B[(B.arm=="hamming_p")&(B.detector==det)].set_index("setting")
    mm=[g.loc[f"p={p}","mean"] for p in ps]
    axes[2].errorbar(ps,mm,yerr=[[g.loc[f"p={p}","mean"]-g.loc[f"p={p}","ci_lo"] for p in ps],[g.loc[f"p={p}","ci_hi"]-g.loc[f"p={p}","mean"] for p in ps]],
                     marker=mk,color=c,capsize=3.5,label=nm)
axes[2].set_xticks(ps); axes[2].set_xlabel("Hamming block exponent $p$"); axes[2].set_ylabel("$P_E$")
axes[2].set_ylim(0.40,0.50); axes[2].legend(frameon=False,loc="lower right"); axes[2].set_title("Block exponent: detectability, 0.05 bpc")
save(fig,"F4_sensitivity")

# ---------------------------------------------------------------- FIG 7 ----
#  Round 3 (R1-1): learned detector over three training seeds, 0.20 bpc
CU=pd.concat([pd.read_csv(f"{D}/K6d_curves_seed{sd}.csv").assign(seed=sd) for sd in (2026,2027,2028)])
SC={2026:CB[0],2027:CB[1],2028:CB[2]}
fig,axes=plt.subplots(1,3,figsize=(16.2,4.5))
for ax,(m,ttl) in zip(axes[:2],(("ours_xtea_d4","Proposed (XTEA) $d_4$: validation $P_E$"),
                              ("ours_naive_d1","Control 2, CFA LSBM $d_1$: validation $P_E$"))):
    for sd in (2026,2027,2028):
        g=CU[(CU.seed==sd)&(CU.method==m)]
        att=list(g.attempt.unique())
        for k,a_ in enumerate(att):
            h=g[g.attempt==a_].sort_values("epoch")
            last=(k==len(att)-1)
            ax.plot(h.epoch,h.val_P_E,color=SC[sd],lw=2.0 if last and len(h)>10 else 1.2,
                    ls="-" if len(h)>10 else ":",label=None)
    ax.axhline(0.5,color="grey",ls=":",lw=1.0)
    if m=="ours_xtea_d4": ax.axhline(0.325,color=CB[6],ls="--",lw=1.1); ax.text(2,0.317,"SPAM-686, $P_E$ = 0.325",fontsize=10,va="top")
    ax.set_ylim(0.25,0.52); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation $P_E$"); ax.set_title(ttl,fontsize=13)
    from matplotlib.lines import Line2D
    hs=[Line2D([0],[0],color=SC[sd],lw=2) for sd in (2026,2027,2028)]+[Line2D([0],[0],color="k",lw=1.2,ls=":")]
    ax.legend(hs,["seed 2026","seed 2027","seed 2028","abandoned attempt"],frameon=False,loc="lower left" if m!="ours_xtea_d4" else "upper right",fontsize=10)
for sd in (2026,2027,2028):
    g=CU[(CU.seed==sd)&(CU.method=="ours_xtea_d4")]
    for k,a_ in enumerate(g.attempt.unique()):
        h=g[g.attempt==a_].sort_values("epoch")
        axes[2].plot(h.epoch,h.train_loss,color=SC[sd],lw=2.0 if len(h)>10 else 1.2,ls="-" if len(h)>10 else ":",
                     label=None)
axes[2].axhline(np.log(2),color="grey",ls=":",lw=1.2); axes[2].text(60,np.log(2)+0.008,"ln 2",fontsize=10.5,color="grey")
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Training cross-entropy"); axes[2].set_title("Proposed (XTEA) $d_4$: training loss",fontsize=13)
axes[2].legend(hs,["seed 2026","seed 2027","seed 2028","abandoned attempt"],frameon=False,loc="upper right",fontsize=10)
save(fig,"F7_cnn_curves")

# ---------------------------------------------------------------- FIG 5 ----
K4=pd.read_csv(f"{D}/K4_summary.csv")
lab=[f"$d_{r.embed_kernel[-1]}\\!\\to\\!d_{r.verify_kernel[-1]}$" for _,r in K4.iterrows()]
x=np.arange(len(K4)); w=0.38
fig,ax=plt.subplots(figsize=(9.6,4.3))
ax.bar(x-w/2,K4.cover_viol_mean,width=w,color=CB[5],label="Cover")
ax.bar(x+w/2,K4.stego_viol_mean,width=w,color=CB[1],label="Stego")
for i,r in K4.reset_index().iterrows():
    ax.annotate(f"AUC\n{r.AUC:.3f}",(i,max(r.cover_viol_mean,r.stego_viol_mean)+0.03),
                ha="center",fontsize=10.5,color="0.25")
    if r.matched: ax.annotate("0",(i,0.012),ha="center",fontsize=11,color="0.25")
ax.set_xticks(x); ax.set_xticklabels(lab)
ax.set_xlabel("Development kernel $\\to$ kernel assumed by the verifier")
ax.set_ylabel("Violation rate"); ax.set_ylim(0,1.06)
ax.legend(frameon=False,loc="upper left",ncol=2)
save(fig,"F5_mismatch")
