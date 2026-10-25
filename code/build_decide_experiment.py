"""
Decide (#6) — synthetic screening experiment, reproducible.
Generates the data (fixed seed), saves decide_data.npz, prints the full stats
table, and regenerates decide_thresholds.png and decide_curve.png from that data.

Target setup: 4,000 patients, ~12% prevalence, AUROC ~0.84, probabilities
calibrated by construction (p is the exact Bayes posterior of the generative model).
Re-run anytime:  python3 build_decide_experiment.py
"""
import numpy as np, os
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

SEED = 20260423
N    = 4000
PI   = 0.12       # target prevalence
D    = 1.41       # class separation -> AUROC ~ Phi(D/sqrt2) ~ 0.84

rng = np.random.default_rng(SEED)
y = (rng.random(N) < PI).astype(int)
# scores: positives ~ N(D,1), negatives ~ N(0,1)
score = rng.normal(D * y, 1.0)
# calibrated posterior p(y=1 | score) under the known generative model (sqrt(2pi) cancels)
f1 = np.exp(-0.5 * (score - D) ** 2)
f0 = np.exp(-0.5 * (score) ** 2)
p = PI * f1 / (PI * f1 + (1 - PI) * f0)

prev = y.mean()
auc  = roc_auc_score(y, p)
n_pos = int(y.sum())

def stats(t):
    pred = (p >= t).astype(int)
    TP = int(((pred == 1) & (y == 1)).sum()); FP = int(((pred == 1) & (y == 0)).sum())
    FN = int(((pred == 0) & (y == 1)).sum()); TN = int(((pred == 0) & (y == 0)).sum())
    sens = TP / (TP + FN) if (TP + FN) else 0.0
    spec = TN / (TN + FP) if (TN + FP) else 0.0
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    return dict(t=t, TP=TP, FP=FP, FN=FN, TN=TN, sens=sens, spec=spec, prec=prec)

# Youden on a fine grid
ts = np.linspace(0.01, 0.99, 197)
sens_arr = np.array([stats(t)["sens"] for t in ts])
spec_arr = np.array([stats(t)["spec"] for t in ts])
J = sens_arr + spec_arr - 1
t_y = float(ts[np.argmax(J)])

np.savez("decide_data.npz", y=y, p=p, prev=prev, auc=auc)

print("="*64)
print(f"N={N}  prevalence={prev*100:.1f}%  ({n_pos} cases)  AUROC={auc:.3f}")
print(f"Youden threshold = {t_y:.3f}")
print("="*64)
hdr = f"{'thresh':>8} {'caught%':>8} {'TP':>5} {'FP':>6} {'FN':>5} {'sens':>6} {'spec':>6} {'prec':>6}"
print(hdr); print("-"*len(hdr))
rows = [0.50, round(t_y,2), 0.20, 0.10]
for t in rows:
    s = stats(t)
    print(f"{t:>8.2f} {s['sens']*100:>7.1f}% {s['TP']:>5} {s['FP']:>6} {s['FN']:>5} "
          f"{s['sens']:>6.3f} {s['spec']:>6.3f} {s['prec']:>6.3f}")
print("-"*len(hdr))
print("cost-ratio map:  R=9 -> p_t=0.10 ,  R=4 -> p_t=0.20 ,  R=1 -> p_t=0.50")

# Net benefit / DCA dominance check
def nb(pt):
    pred = (p >= pt).astype(int)
    TP = ((pred == 1) & (y == 1)).sum(); FP = ((pred == 1) & (y == 0)).sum()
    w = pt / (1 - pt)
    return TP / N - FP / N * w, prev - (1 - prev) * w  # (model, treat-all)
pts = np.linspace(0.01, 0.40, 120)
nbm = np.array([nb(pt)[0] for pt in pts]); nba = np.array([nb(pt)[1] for pt in pts])
beats_all  = bool(np.all(nbm >= nba - 1e-9))
beats_none = bool(np.all(nbm >= -1e-9))
# where (if anywhere) the model dips below the better of the two defaults
best_default = np.maximum(nba, 0)
below = pts[nbm < best_default - 1e-9]
print("="*64)
print(f"DCA over thresholds 0.01-0.40:  model >= treat-all everywhere: {beats_all}")
print(f"                                model >= treat-none everywhere: {beats_none}")
if below.size:
    print(f"   model falls below the better default at thresholds: "
          f"{below.min():.3f}-{below.max():.3f}")
else:
    print("   model dominates both defaults across the entire range.")
print("="*64)

# ---------------- figures (house style) ----------------
FD="/usr/share/fonts/opentype/inter/"
for fn in ["Inter-Regular.otf","Inter-Medium.otf","Inter-SemiBold.otf","Inter-Bold.otf","Inter-ExtraBold.otf"]:
    if os.path.exists(FD+fn): fm.fontManager.addfont(FD+fn)
Rf=fm.FontProperties(fname=FD+"Inter-Regular.otf");Mf=fm.FontProperties(fname=FD+"Inter-Medium.otf")
SBf=fm.FontProperties(fname=FD+"Inter-SemiBold.otf");Bf=fm.FontProperties(fname=FD+"Inter-Bold.otf")
XBf=fm.FontProperties(fname=FD+"Inter-ExtraBold.otf")
NAVY="#12285A";TEAL="#22B2A0";VIOLET="#8E7DEC";MUTED="#5A6472";GRID="#E7E0CE"
WHITE="#FFFFFF";TEAL_DK="#0E7C6F";VIOLET_DK="#6C4FD0";RED="#C64B4B";PAPER="#F4EEE2"

# FIG A: sensitivity vs specificity across thresholds
fig,ax=plt.subplots(figsize=(8,5.0),dpi=170); fig.patch.set_facecolor(PAPER); ax.set_facecolor(WHITE)
ax.plot(ts,sens_arr,color=TEAL,lw=2.6,label="Sensitivity (cases caught)")
ax.plot(ts,spec_arr,color=VIOLET,lw=2.6,label="Specificity (healthy left alone)")
for tv,c in [(0.5,RED),(t_y,NAVY),(0.10,TEAL_DK)]:
    ax.axvline(tv,color=c,ls=(0,(3,3)),lw=1.4,alpha=0.9)
s5=stats(0.5)
ax.annotate(f"at 0.5, only {s5['sens']*100:.0f}% of cases caught\n({s5['TP']} of {n_pos}) — {s5['FN']} missed",
            xy=(0.5,s5['sens']),xytext=(0.60,0.34),fontproperties=SBf,fontsize=10.5,color=RED,
            arrowprops=dict(arrowstyle="-|>",color=RED,lw=1.4,connectionstyle="arc3,rad=-0.2"))
ax.text(0.10-0.020,0.30,"Cost 9:1  → 0.10",rotation=90,ha="right",va="center",fontproperties=Bf,fontsize=9.5,color=TEAL_DK)
ax.text(t_y+0.014,0.28,f"Youden {t_y:.2f}",rotation=90,ha="left",va="center",fontproperties=Bf,fontsize=9.5,color=NAVY)
ax.text(0.5+0.014,0.66,"Default 0.5",rotation=90,ha="left",va="center",fontproperties=Bf,fontsize=9.5,color=RED)
ax.set_xlim(0,1);ax.set_ylim(0,1)
ax.set_xlabel("Decision threshold — act if predicted risk ≥ this",fontproperties=Bf,fontsize=12,color=NAVY,labelpad=7)
ax.set_ylabel("Rate",fontproperties=Bf,fontsize=12,color=NAVY,labelpad=7)
for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(Mf); t.set_color(MUTED)
ax.tick_params(labelsize=9); ax.grid(True,color=GRID,lw=0.8)
[ax.spines[s].set_visible(False) for s in ["top","right"]]; [ax.spines[s].set_color(GRID) for s in ["left","bottom"]]
ax.legend(loc="lower left",frameon=False,prop=Mf,fontsize=10)
fig.text(0.055,0.955,"One model, three thresholds — the cutoff is the decision",fontproperties=XBf,fontsize=13.5,color=NAVY)
fig.text(0.965,0.028,f"synthetic screening model · prevalence {prev*100:.0f}% · AUROC {auc:.2f} · gobeyondaccuracy.com",
         ha="right",fontproperties=Mf,fontsize=8,color=MUTED)
plt.subplots_adjust(left=0.10,right=0.965,top=0.86,bottom=0.16)
fig.savefig("decide_thresholds.png",facecolor=PAPER,dpi=170); plt.close(fig)

# FIG B: decision curve
fig,ax=plt.subplots(figsize=(8,5.0),dpi=170); fig.patch.set_facecolor(PAPER); ax.set_facecolor(WHITE)
ax.axhline(0,color=MUTED,lw=1.6)
ax.plot(pts,nba,color=VIOLET,lw=2.2,ls=(0,(5,3)))
ax.plot(pts,nbm,color=NAVY,lw=3)
ax.text(0.052,0.086,"refer everyone",fontproperties=Bf,fontsize=10.5,color=VIOLET_DK,ha="left",va="bottom")
ax.fill_between(pts,np.maximum(nba,0),nbm,where=(nbm>np.maximum(nba,0)),color=TEAL,alpha=0.16,lw=0)
ax.text(0.052,0.968,"Does acting on the model actually help?",transform=ax.transAxes,fontproperties=XBf,fontsize=13.5,color=NAVY)
ax.text(0.052,0.905,"Net benefit vs. referring everyone or no one, across sensible thresholds.",
        transform=ax.transAxes,fontproperties=Mf,fontsize=10,color=MUTED)
ax.annotate("the model leads both\nacross sensible thresholds",xy=(0.24,nb(0.24)[0]),xytext=(0.255,0.062),
            fontproperties=SBf,fontsize=11,color=TEAL_DK,
            arrowprops=dict(arrowstyle="-|>",color=TEAL_DK,lw=1.4,connectionstyle="arc3,rad=0.2"))
ax.text(0.35,-0.03,"refer everyone → net benefit below zero",fontproperties=SBf,fontsize=9.5,color=VIOLET_DK,ha="right")
ax.text(0.383,0.002,"refer no one",fontproperties=SBf,fontsize=9.5,color=MUTED,ha="right",va="bottom")
ax.set_xlim(0.01,0.40);ax.set_ylim(-0.06,0.11)
ax.set_xlabel("Threshold probability — how sure before you act (= the harm ratio)",fontproperties=Bf,fontsize=12,color=NAVY,labelpad=7)
ax.set_ylabel("Net benefit",fontproperties=Bf,fontsize=12,color=NAVY,labelpad=7)
for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(Mf); t.set_color(MUTED)
ax.tick_params(labelsize=9); ax.grid(True,color=GRID,lw=0.8)
[ax.spines[s].set_visible(False) for s in ["top","right"]]; [ax.spines[s].set_color(GRID) for s in ["left","bottom"]]
fig.text(0.965,0.028,"Decision Curve Analysis (Vickers & Elkin, 2006) · gobeyondaccuracy.com",ha="right",fontproperties=Mf,fontsize=8,color=MUTED)
plt.subplots_adjust(left=0.10,right=0.965,top=0.99,bottom=0.16)
fig.savefig("decide_curve.png",facecolor=PAPER,dpi=170); plt.close(fig)
print("wrote decide_data.npz, decide_thresholds.png, decide_curve.png")
