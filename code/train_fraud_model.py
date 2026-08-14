"""
train_fraud_model.py — reproduce the real numbers used in the F1 and AUROC chapters.

Trains a plain logistic-regression model on the ULB credit-card fraud benchmark
and prints the exact figures the handbook cites:
  - confusion matrix at threshold 0.5:  TP=91, FP=16, FN=57, TN=85,279
  - precision 0.85, recall 0.615, F1 0.714
  - AUROC ≈ 0.957, PR-AUC ≈ 0.71
  - a few operating points along the SAME model's ROC curve
It also saves the ROC figure to outputs/roc_curve.png.

Data: download data/creditcard.csv first (see data/README.md).
Run from the repo root:  python code/train_fraud_model.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score,
                             average_precision_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "creditcard.csv"
OUT = ROOT / "outputs"; OUT.mkdir(exist_ok=True)

def main():
    if not DATA.exists():
        raise SystemExit(f"Dataset not found at {DATA}\nSee data/README.md to download it from Kaggle.")

    df = pd.read_csv(DATA)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    scores = clf.predict_proba(Xte)[:, 1]

    # confusion matrix at the default 0.5 cutoff
    pred = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(yte, pred).ravel()]
    print(f"Test set: {len(yte):,} transactions, {int(yte.sum())} fraud\n")
    print("Confusion matrix @ threshold 0.5")
    print(f"  TP (caught)          {tp}")
    print(f"  FP (false alarms)    {fp}")
    print(f"  FN (missed)          {fn}")
    print(f"  TN (cleared)         {tn:,}\n")
    print(f"  accuracy   {accuracy_score(yte, pred):.4f}")
    print(f"  precision  {precision_score(yte, pred):.3f}")
    print(f"  recall     {recall_score(yte, pred):.3f}")
    print(f"  F1         {f1_score(yte, pred):.3f}\n")
    print(f"  AUROC      {roc_auc_score(yte, scores):.4f}   (threshold-independent)")
    print(f"  PR-AUC     {average_precision_score(yte, scores):.4f}   (the honest number under imbalance)\n")

    # operating points along the SAME model's ROC — slide the threshold
    fpr, tpr, thr = roc_curve(yte, scores)
    neg = int((yte == 0).sum())
    print("Slide the threshold on this one model:")
    for target in (0.615, 0.80, 0.88):
        i = int(np.argmin(np.abs(tpr - target)))
        print(f"  recall ~{tpr[i]:.2f}  ->  {int(round(fpr[i]*neg)):>5} false alarms  (threshold {thr[i]:.4g})")

    # ROC figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        NAVY, TEAL, MUTED = "#12285A", "#0F766E", "#5A6472"
        fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
        ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.5)
        ax.plot(fpr, tpr, color=TEAL, lw=2.5)
        ax.fill_between(fpr, tpr, color=TEAL, alpha=0.10)
        ax.set_xlabel("False positive rate (legit flagged)")
        ax.set_ylabel("True positive rate = recall (fraud caught)")
        ax.set_title(f"ROC curve — AUROC = {roc_auc_score(yte, scores):.3f}", color=NAVY)
        fig.tight_layout()
        fig.savefig(OUT / "roc_curve.png")
        print(f"\nSaved ROC figure -> {OUT / 'roc_curve.png'}")
    except ImportError:
        print("\n(matplotlib not installed — skipped the ROC figure)")

if __name__ == "__main__":
    main()
