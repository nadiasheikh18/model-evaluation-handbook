"""
Calibration on real data (Wisconsin breast-cancer, WDBC) — reproducible.
Reproduces every number and figure in handbook/03-calibration.md.

Leakage controls (so the numbers are honest):
  * Out-of-fold probabilities: cross_val_predict with a Pipeline, so the scaler
    and the model are fit only on each training fold, never on the row being
    scored. The reported accuracy / ECE / confidence numbers are therefore
    measured on data each model did not train on.
  * Independent recalibration test: the temperature is FIT on one half of the
    out-of-fold predictions (the calibration half) and the before/after ECE is
    REPORTED on the other, held-out half — so the "after" number is not measured
    on the data used to choose the temperature.
  * Config is fixed and explicit (SEED, N_BINS, CONF_THR, CV below), so the exact
    values printed here are the values quoted in the chapter.

Run:  python code/calibration_analysis_wdbc.py
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict, StratifiedKFold

SEED = 0            # all randomness
N_BINS = 10         # calibration bins on [0.5, 1.0] (top-label confidence)
CONF_THR = 0.80     # "high confidence" threshold
CV = 10             # cross-validation folds

d = load_breast_cancer()
X, y, names = d.data, d.target, list(d.feature_names)
# A deliberately weak 10-feature subset, so the task carries genuine uncertainty
# (a full-feature model on WDBC is nearly perfect and calibration is uninteresting).
use = ['mean texture','mean smoothness','mean symmetry','mean fractal dimension',
       'texture error','smoothness error','symmetry error','fractal dimension error',
       'worst texture','worst smoothness']
idx = [names.index(u) for u in use]; Xs = X[:, idx]
print(f"real WDBC: {X.shape[0]} patients, {len(idx)} of 30 features, {CV}-fold OOF, seed={SEED}")

cvsplit = StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED)
def oof(model):
    return cross_val_predict(model, Xs, y, cv=cvsplit, method="predict_proba")[:, 1]

p_log = oof(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))
p_mlp = oof(make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,32), max_iter=1500, random_state=SEED)))
p_gnb = oof(make_pipeline(StandardScaler(), GaussianNB()))

def ece(p, yy, nb=N_BINS):
    conf = np.maximum(p, 1-p); pred = (p >= 0.5).astype(int); corr = (pred == yy).astype(int)
    edges = np.linspace(0.5, 1, nb+1); e = 0.0
    for i in range(nb):
        lo, hi = edges[i], edges[i+1]
        m = (conf >= lo) & (conf <= hi if i == nb-1 else conf < hi)
        if m.sum() == 0: continue
        e += (m.sum()/len(conf)) * abs(corr[m].mean() - conf[m].mean())
    return e

def summary(p):
    conf = np.maximum(p, 1-p); pred = (p >= 0.5).astype(int)
    corr = (pred == y).astype(int); wrong = corr == 0; hi = conf > CONF_THR
    return dict(
        acc          = corr.mean()*100,
        ece          = ece(p, y),
        meanconf     = conf.mean()*100,
        frac_hi      = hi.mean()*100,                                   # share of ALL preds that are high-conf
        share_err_hi = (wrong & hi).sum()/max(wrong.sum(),1)*100,       # of the errors, share made at high conf
        err_rate_hi  = wrong[hi].mean()*100 if hi.sum() else 0.0,       # among high-conf preds, share WRONG
        acc_hi       = corr[hi].mean()*100 if hi.sum() else 0.0,        # accuracy in the high-conf region
        meanconf_hi  = conf[hi].mean()*100 if hi.sum() else 0.0,        # stated confidence there
    )

models = [("LogReg", p_log), ("MLP", p_mlp), ("GaussianNB", p_gnb)]
S = {nm: summary(p) for nm, p in models}
print("\nmodel        acc   ECE   meanconf  %preds>.8  %errs>.8  errRate>.8  acc>.8  conf>.8")
for nm, _ in models:
    s = S[nm]
    print(f"{nm:11s} {s['acc']:4.0f}% {s['ece']:.3f}   {s['meanconf']:4.0f}%     {s['frac_hi']:4.0f}%     "
          f"{s['share_err_hi']:4.0f}%     {s['err_rate_hi']:4.0f}%     {s['acc_hi']:4.0f}%  {s['meanconf_hi']:4.0f}%")

# honest = lowest ECE, overconfident = highest ECE
honest_nm  = min(S, key=lambda k: S[k]['ece'])
over_nm    = max(S, key=lambda k: S[k]['ece'])
p_honest   = dict(models)[honest_nm]; over_p = dict(models)[over_nm]
print(f"\nhonest = {honest_nm} (ECE {S[honest_nm]['ece']:.3f}) | overconfident = {over_nm} (ECE {S[over_nm]['ece']:.3f})")

# ---- temperature scaling, evaluated INDEPENDENTLY (fit on calib half, report on test half) ----
def logit(p): p = np.clip(p, 1e-6, 1-1e-6); return np.log(p/(1-p))
rng = np.random.default_rng(SEED)
order = rng.permutation(len(y)); half = len(y)//2
cal, tst = order[:half], order[half:]
z = logit(over_p)
def nll_T(T, ii):
    q = 1/(1+np.exp(-z[ii]/T)); q = np.clip(q, 1e-9, 1-1e-9)
    return -(y[ii]*np.log(q) + (1-y[ii])*np.log(1-q)).mean()
Ts = np.linspace(0.5, 4, 351); Tb = Ts[np.argmin([nll_T(T, cal) for T in Ts])]
p_temp = 1/(1+np.exp(-z/Tb))
ece_before = ece(over_p[tst], y[tst]); ece_after = ece(p_temp[tst], y[tst])
acc_before = ((over_p[tst] >= 0.5).astype(int) == y[tst]).mean()*100
acc_after  = ((p_temp[tst]  >= 0.5).astype(int) == y[tst]).mean()*100
print(f"\ntemperature scaling on {over_nm}:  T={Tb:.2f}  (fit on calibration half, reported on held-out test half)")
print(f"   test-half ECE  {ece_before:.3f} -> {ece_after:.3f}   test-half accuracy {acc_before:.0f}% -> {acc_after:.0f}% (unchanged)")

# ---------------- figures (house style) ----------------
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = "/usr/share/fonts/opentype/inter/"
for fn in ["Inter-Regular.otf","Inter-Medium.otf","Inter-SemiBold.otf","Inter-Bold.otf","Inter-ExtraBold.otf"]:
    if os.path.exists(FD+fn): fm.fontManager.addfont(FD+fn)
def _f(n): return fm.FontProperties(fname=FD+n) if os.path.exists(FD+n) else fm.FontProperties()
Mf=_f("Inter-Medium.otf"); SBf=_f("Inter-SemiBold.otf"); Bf=_f("Inter-Bold.otf"); XBf=_f("Inter-ExtraBold.otf")
NAVY="#12285A"; TEAL="#22B2A0"; VIOLET="#8E7DEC"; MUTED="#5A6472"; GRID="#E7E0CE"
WHITE="#FFFFFF"; TEAL_DK="#0E7C6F"; RED="#C64B4B"; AMBER="#C98A2B"; PAPER="#F4EEE2"

def rel_curve(p, nb=8):
    conf=np.maximum(p,1-p); pred=(p>=0.5).astype(int); corr=(pred==y).astype(int)
    edges=np.linspace(0.5,1,nb+1); xs=[]; ys=[]
    for i in range(nb):
        lo,hi=edges[i],edges[i+1]; m=(conf>=lo)&(conf<=hi if i==nb-1 else conf<hi)
        if m.sum()<12: continue
        xs.append(conf[m].mean()); ys.append(corr[m].mean())
    return np.array(xs), np.array(ys)

# FIG A: reliability — honest vs overconfident
xh,yh = rel_curve(p_honest); xo,yo = rel_curve(over_p)
fig,ax=plt.subplots(figsize=(8,5.2),dpi=170); fig.patch.set_facecolor(PAPER); ax.set_facecolor(WHITE)
ax.plot([0.5,1],[0.5,1],color=MUTED,ls=(0,(3,3)),lw=1.4)
ax.text(0.90,0.945,"perfect calibration",rotation=33,fontproperties=Mf,fontsize=9,color=MUTED,ha="center")
ax.plot(xh,yh,color=TEAL,lw=2.8,marker="o",ms=6,label=f"{honest_nm} — lower ECE ({S[honest_nm]['ece']:.2f})")
ax.plot(xo,yo,color=RED,lw=2.8,marker="s",ms=6,label=f"{over_nm} — higher ECE ({S[over_nm]['ece']:.2f})")
ax.set_xlim(0.5,1); ax.set_ylim(0.3,1)
ax.set_xlabel("Confidence the model states",fontproperties=Bf,fontsize=12,color=NAVY,labelpad=7)
ax.set_ylabel("Accuracy it actually delivers",fontproperties=Bf,fontsize=12,color=NAVY,labelpad=7)
for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(Mf); t.set_color(MUTED)
ax.tick_params(labelsize=9); ax.grid(True,color=GRID,lw=0.8)
[ax.spines[s].set_visible(False) for s in ["top","right"]]; [ax.spines[s].set_color(GRID) for s in ["left","bottom"]]
ax.legend(loc="upper left",frameon=False,prop=Mf,fontsize=10)
ax.annotate("in the high-confidence range,\nsits below the line",xy=(xo[-1],yo[-1]),xytext=(0.72,0.52),
            fontproperties=SBf,fontsize=10.5,color=RED,
            arrowprops=dict(arrowstyle="-|>",color=RED,lw=1.3,connectionstyle="arc3,rad=0.2"))
fig.text(0.055,0.955,"Two models, same task — their stated confidence lines up with reality differently",fontproperties=XBf,fontsize=12,color=NAVY)
fig.text(0.965,0.028,"real breast-cancer data (WDBC, 569 patients, out-of-fold) · gobeyondaccuracy.com",ha="right",fontproperties=Mf,fontsize=8,color=MUTED)
plt.subplots_adjust(left=0.10,right=0.965,top=0.88,bottom=0.15)
fig.savefig("fig_calibration_reliability.png",facecolor=PAPER,dpi=170); plt.close(fig)

# FIG B: left = accuracy; right = high-confidence region, stated confidence vs actual accuracy
fig,(axL,axR)=plt.subplots(1,2,figsize=(10.4,5.0),dpi=170); fig.patch.set_facecolor(PAPER)
labs=[nm for nm,_ in models]; cols=[TEAL,RED,AMBER]
for a in (axL,axR):
    a.set_facecolor(WHITE); a.grid(True,axis="y",color=GRID,lw=0.8)
    [a.spines[s].set_visible(False) for s in ["top","right"]]; [a.spines[s].set_color(GRID) for s in ["left","bottom"]]
    for t in a.get_xticklabels(): t.set_fontproperties(SBf); t.set_color(NAVY); t.set_fontsize(10)
    for t in a.get_yticklabels(): t.set_fontproperties(Mf); t.set_color(MUTED); t.set_fontsize(9)
accs=[S[nm]['acc'] for nm in labs]
b=axL.bar(labs,accs,color=cols,width=0.62)
for bi,v in zip(b,accs): axL.text(bi.get_x()+bi.get_width()/2,v+1.2,f"{v:.0f}%",ha="center",fontproperties=Bf,fontsize=12,color=NAVY)
axL.set_ylim(0,100); axL.set_ylabel("Overall accuracy",fontproperties=Bf,fontsize=11.5,color=NAVY,labelpad=6)
axL.set_title("The most accurate model…",fontproperties=Bf,fontsize=11.5,color=NAVY,pad=8)
# right: among predictions it made at >80% confidence, stated vs actual
import numpy as _np
x=_np.arange(len(labs)); w=0.36
stated=[S[nm]['meanconf_hi'] for nm in labs]; actual=[S[nm]['acc_hi'] for nm in labs]
b1=axR.bar(x-w/2,stated,width=w,color=MUTED,label="confidence it states")
b2=axR.bar(x+w/2,actual,width=w,color=cols[1],label="accuracy it delivers")
for bars_,vals_ in [(b1,stated),(b2,actual)]:
    for bi,v in zip(bars_,vals_): axR.text(bi.get_x()+bi.get_width()/2,v+1.0,f"{v:.0f}%",ha="center",fontproperties=Bf,fontsize=9.5,color=NAVY)
axR.set_xticks(x); axR.set_xticklabels(labs)
axR.set_ylim(0,110); axR.set_ylabel("Within its >80%-confidence predictions",fontproperties=Bf,fontsize=10.5,color=NAVY,labelpad=6)
axR.legend(loc="lower center",frameon=False,prop=Mf,fontsize=9,ncol=2)
axR.set_title("…overstates its confidence where it's most sure",fontproperties=Bf,fontsize=11.5,color=NAVY,pad=8)
fig.text(0.5,0.028,"Accuracy and calibration are different axes. Right panel (leakage-free): within each model's own >80%-confidence predictions, stated confidence vs. delivered accuracy.",
         ha="center",fontproperties=Mf,fontsize=8.5,color=MUTED)
plt.subplots_adjust(left=0.075,right=0.975,top=0.86,bottom=0.17,wspace=0.24)
fig.savefig("fig_calibration_overconfidence.png",facecolor=PAPER,dpi=170); plt.close(fig)
print("\nwrote fig_calibration_reliability.png, fig_calibration_overconfidence.png")
