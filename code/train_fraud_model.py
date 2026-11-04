"""
train_fraud_model.py — rebuild every number and figure the F1 (ch.1) and AUROC
(ch.2) chapters cite, from a clean checkout.

It reproduces, on the ULB credit-card fraud benchmark:
  PLAIN logistic regression  @0.5 : TP=91  FP=16   FN=57 TN=85,279
                                     acc 99.91%  precision 0.85  recall 0.615  F1 0.714
                                     AUROC 0.957   Average Precision (AP) 0.71
  DO-NOTHING (predict all legit)   : recall 0.000  F1 0.000  acc 99.83%
  AGGRESSIVE (class_weight=balanced): recall 0.878  precision 0.067  F1 0.125
                                      130 of 148 caught, 1,807 false alarms
  Operating points on the PLAIN model's ROC (slide the threshold):
     62% caught -> 16 FP ,  80% caught -> ~77 FP ,  88% caught -> 5,301 FP

IMPORTANT — two splits, on purpose:
  * RANDOM stratified split  = the chapter's *illustrative* numbers. A random split on
    time-ordered transactions leaks future information, so these are a teaching benchmark,
    NOT an operational estimate (the chapter itself warns about this; see ch.1).
  * CHRONOLOGICAL split on Time = the honest operational estimate (train on the past,
    test on the future). Recall at 0.5 drops (~0.615 -> ~0.49): the leak, made visible.

Figures (house style, written to handbook/figures/):
  fig_f1_confusion_matrix.png , fig_auroc_roc_curve.png
Note: fig_auroc_ranking.png is a hand-drawn conceptual illustration, not produced here
(see RESULTS_MANIFEST.md).

Data: download data/creditcard.csv first (see data/README.md).
Run from anywhere:  python code/train_fraud_model.py
                    python code/train_fraud_model.py --data /path/to/creditcard.csv
"""
import os, argparse, platform
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score, average_precision_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]

def fit(Xtr, Xte, ytr, yte, balanced=False, seed=42):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, random_state=seed,
                             class_weight=("balanced" if balanced else None)).fit(sc.transform(Xtr), ytr)
    s = clf.predict_proba(sc.transform(Xte))[:, 1]
    return s

def report(yte, s, label):
    pred = (s >= 0.5).astype(int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(yte, pred).ravel()]
    print(f"  [{label:28}] TP={tp:>4} FP={fp:>5} FN={fn:>3} TN={tn:>7,} | "
          f"acc={accuracy_score(yte,pred)*100:6.2f}% prec={precision_score(yte,pred,zero_division=0):.3f} "
          f"rec={recall_score(yte,pred):.3f} F1={f1_score(yte,pred,zero_division=0):.3f} | "
          f"AUROC={roc_auc_score(yte,s):.4f} AP={average_precision_score(yte,s):.4f}")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, auroc=roc_auc_score(yte, s), ap=average_precision_score(yte, s),
                prec=precision_score(yte, pred, zero_division=0), rec=recall_score(yte, pred),
                f1=f1_score(yte, pred, zero_division=0), acc=accuracy_score(yte, pred))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "creditcard.csv"))
    ap.add_argument("--figdir", default=str(ROOT / "handbook" / "figures"))
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    print(f"versions: python {platform.python_version()}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  scikit-learn {sklearn.__version__}")
    data = Path(a.data)
    if not data.exists():
        raise SystemExit(f"Dataset not found at {data}\nSee data/README.md to download it from Kaggle "
                         f"(or pass --data /path/to/creditcard.csv).")
    figdir = Path(a.figdir); figdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    # ===================== RANDOM split — the chapter's ILLUSTRATIVE numbers =====================
    print("\n" + "="*78)
    print("RANDOM stratified split (seed %d) — ILLUSTRATIVE teaching numbers, NOT operational" % a.seed)
    print("  (a random split on time-ordered data leaks the future; see the chronological split below)")
    print("="*78)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=a.seed)
    print(f"  test set: {len(yte):,} transactions, {int(yte.sum())} fraud\n")
    s_plain = fit(Xtr, Xte, ytr, yte, seed=a.seed)
    plain = report(yte, s_plain, "PLAIN logistic regression")
    s_aggr = fit(Xtr, Xte, ytr, yte, balanced=True, seed=a.seed)
    aggr = report(yte, s_aggr, "AGGRESSIVE (balanced)")
    none_pred = np.zeros_like(yte)
    print(f"  [{'DO-NOTHING (all legit)':28}] TP=   0 FP=    0 FN={int(yte.sum()):>3} TN={int((yte==0).sum()):>7,} | "
          f"acc={accuracy_score(yte,none_pred)*100:6.2f}% prec=0.000 rec=0.000 F1=0.000 | AUROC=0.5000 AP={yte.mean():.4f}")

    print("\n  Accuracy can't rank these; F1 can:")
    print(f"    do-nothing  acc {accuracy_score(yte,none_pred)*100:.2f}%  F1 0.000")
    print(f"    plain       acc {plain['acc']*100:.2f}%  F1 {plain['f1']:.3f}")
    print(f"    aggressive  acc {aggr['acc']*100:.2f}%  F1 {aggr['f1']:.3f}")
    print(f"    (aggressive catches MORE fraud than plain yet scores a LOWER F1 — recall bought with a flood of FPs)")

    print("\n  Operating points on the PLAIN model's ROC (same model, slide the threshold):")
    fpr, tpr, thr = roc_curve(yte, s_plain); neg = int((yte == 0).sum())
    ops = []
    for tgt in (0.615, 0.80, 0.88):
        i = int(np.argmin(np.abs(tpr - tgt)))
        fpc = int(round(fpr[i] * neg))
        ops.append((tpr[i], fpc, thr[i]))
        print(f"     recall ~{tpr[i]*100:4.1f}%  ->  {fpc:>5} false alarms  (threshold {thr[i]:.4g})")

    # ===================== CHRONOLOGICAL split — the HONEST operational estimate =====================
    print("\n" + "="*78)
    print("CHRONOLOGICAL split on Time (first 70% train / last 30% test) — operational estimate")
    print("="*78)
    order = np.argsort(df["Time"].values, kind="mergesort"); cut = int(0.70 * len(order))
    tr, te = order[:cut], order[cut:]
    print(f"  test set: {len(te):,} transactions, {int(y[te].sum())} fraud (fraud is not uniform over time)\n")
    s_chrono = fit(X[tr], X[te], y[tr], y[te], seed=a.seed)
    chrono = report(y[te], s_chrono, "PLAIN (chronological)")
    print("  -> recall at 0.5 drops vs the random split: the future-leak the chapter warns about, made visible.")

    # ===================== figures (house style, graceful font fallback) =====================
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    FD = "/usr/share/fonts/opentype/inter/"
    for fn in ["Inter-Regular.otf","Inter-Medium.otf","Inter-SemiBold.otf","Inter-Bold.otf","Inter-ExtraBold.otf"]:
        if os.path.exists(FD+fn): fm.fontManager.addfont(FD+fn)
    def _f(n): return fm.FontProperties(fname=FD+n) if os.path.exists(FD+n) else fm.FontProperties()
    Rf,Mf,SBf,Bf,XBf = _f("Inter-Regular.otf"),_f("Inter-Medium.otf"),_f("Inter-SemiBold.otf"),_f("Inter-Bold.otf"),_f("Inter-ExtraBold.otf")
    NAVY="#12285A";TEAL="#22B2A0";TEAL_DK="#0E7C6F";MUTED="#5A6472";GRID="#E7E0CE";WHITE="#FFFFFF";RED="#C64B4B";PAPER="#F4EEE2"

    # FIG: confusion matrix (plain model, random split, @0.5)
    cells=[[plain['tp'],plain['fn']],[plain['fp'],plain['tn']]]
    labels=[["caught (TP)","missed (FN)"],["false alarm (FP)","cleared (TN)"]]
    colors=[[TEAL,RED],[RED,WHITE]]
    fig,ax=plt.subplots(figsize=(7.2,4.6),dpi=170); fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER); ax.axis("off")
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c,1-r),1,1,fc=colors[r][c],ec=GRID,lw=1.5,alpha=0.22 if colors[r][c]!=WHITE else 1.0))
            ax.add_patch(plt.Rectangle((c,1-r),1,1,fc="none",ec=GRID,lw=1.5))
            ax.text(c+0.5,1-r+0.60,f"{cells[r][c]:,}",ha="center",va="center",fontproperties=XBf,fontsize=22,color=NAVY)
            ax.text(c+0.5,1-r+0.26,labels[r][c],ha="center",va="center",fontproperties=SBf,fontsize=11,color=MUTED)
    ax.text(0.5,2.14,"predicted FRAUD",ha="center",fontproperties=Bf,fontsize=11,color=TEAL_DK)
    ax.text(1.5,2.14,"predicted LEGIT",ha="center",fontproperties=Bf,fontsize=11,color=MUTED)
    ax.set_xlim(-0.05,2.05);ax.set_ylim(-0.05,2.35)
    fig.text(0.055,0.955,"Plain model at the 0.5 cutoff — 99.91% accurate, catches 61.5% of fraud",fontproperties=XBf,fontsize=12.5,color=NAVY)
    fig.text(0.965,0.03,"real ULB credit-card benchmark · random split (illustrative) · gobeyondaccuracy.com",ha="right",fontproperties=Mf,fontsize=8,color=MUTED)
    plt.subplots_adjust(left=0.03,right=0.97,top=0.88,bottom=0.05)
    fig.savefig(figdir/"fig_f1_confusion_matrix.png",facecolor=PAPER,dpi=170); plt.close(fig)

    # FIG: ROC curve with the two operating points
    fig,ax=plt.subplots(figsize=(7.4,5.4),dpi=170); fig.patch.set_facecolor(PAPER); ax.set_facecolor(WHITE)
    ax.plot([0,1],[0,1],ls=(0,(3,3)),color=MUTED,lw=1.4)
    ax.plot(fpr,tpr,color=TEAL,lw=2.8); ax.fill_between(fpr,tpr,color=TEAL,alpha=0.10,lw=0)
    for rec,fpc,t in [ops[0],ops[2]]:
        xi=fpc/neg
        ax.plot(xi,rec,'o',ms=9,color=NAVY,zorder=5)
        ax.annotate(f"{rec*100:.0f}% caught\n{fpc:,} false alarms",xy=(xi,rec),
                    xytext=(xi+0.12,rec-0.16),fontproperties=SBf,fontsize=10.5,color=NAVY,
                    arrowprops=dict(arrowstyle="-|>",color=NAVY,lw=1.3,connectionstyle="arc3,rad=0.2"))
    ax.set_xlim(0,1);ax.set_ylim(0,1.01)
    ax.set_xlabel("False positive rate (legit flagged)",fontproperties=Bf,fontsize=12,color=NAVY,labelpad=7)
    ax.set_ylabel("True positive rate = recall (fraud caught)",fontproperties=Bf,fontsize=12,color=NAVY,labelpad=7)
    for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontproperties(Mf); t.set_color(MUTED)
    ax.tick_params(labelsize=9); ax.grid(True,color=GRID,lw=0.8)
    [ax.spines[sp].set_visible(False) for sp in ["top","right"]]; [ax.spines[sp].set_color(GRID) for sp in ["left","bottom"]]
    fig.text(0.055,0.955,f"A 0.96-AUROC model still buries you in false alarms  (AUROC {plain['auroc']:.3f})",fontproperties=XBf,fontsize=12.5,color=NAVY)
    fig.text(0.965,0.018,"real ULB benchmark · random split (illustrative) · gobeyondaccuracy.com",ha="right",fontproperties=Mf,fontsize=8,color=MUTED)
    plt.subplots_adjust(left=0.10,right=0.965,top=0.9,bottom=0.17)
    fig.savefig(figdir/"fig_auroc_roc_curve.png",facecolor=PAPER,dpi=170); plt.close(fig)

    print(f"\nwrote figures to {figdir}:  fig_f1_confusion_matrix.png, fig_auroc_roc_curve.png")
    print("AP = Average Precision (what average_precision_score computes); it summarizes the precision-recall")
    print("curve and is often more informative than AUROC under strong class imbalance (Saito & Rehmsmeier, 2015).")

if __name__ == "__main__":
    main()
