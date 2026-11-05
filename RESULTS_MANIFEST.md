# Results manifest — which command rebuilds which number and figure

*So every empirical claim in the handbook is rebuildable from a clean checkout. Run from the repo root.*

## Setup
```
# 1. install (see requirements.txt); 2. get the fraud data:
#    download creditcard.csv from Kaggle (ULB credit-card fraud) into data/
#    (expected filename, byte size, SHA-256, and a schema check are in data/README.md)
```

## F1 & fraud (ch. 1) and AUROC (ch. 2) → `code/train_fraud_model.py`
```
python code/train_fraud_model.py            # or: --data /path/to/creditcard.csv
```
Two splits are produced on purpose:

| Chapter claim | Value | Split | Source |
|---|---|---|---|
| Confusion @0.5 (plain): caught / false alarm / missed / cleared | 91 / 16 / 57 / 85,279 | random (illustrative) | `train_fraud_model.py` → `fig_f1_confusion_matrix.png` |
| Plain model accuracy / precision / recall / F1 | 99.91% / 0.85 / 0.615 / 0.714 | random | same |
| AUROC / Average Precision (AP) | 0.957 / 0.71 | random | same |
| Do-nothing model recall / F1 | 0.000 / 0.000 | random | same |
| Aggressive (class_weight=balanced): recall / precision / F1 | 0.878 / 0.067 / 0.125 | random | same |
| Aggressive: caught / false alarms | 130 of 148 / 1,807 | random | same |
| Three-model F1 (do-nothing / plain / aggressive) | 0.00 / 0.714 / 0.125 | random | same |
| ROC operating points (same plain model, slide threshold) | 62% → 16 FP · 80% → ~77 FP · 88% → 5,301 FP | random | same → `fig_auroc_roc_curve.png` |
| **Chronological temporal-holdout estimate (plain, split on Time)** | recall 0.49, F1 0.60 (vs 0.615/0.714 random) | **chronological** | same |

**Why two splits.** The chapter's headline numbers use a **random** split — fast and standard, but a random split on time-ordered transactions leaks future information, so those figures are an *illustrative teaching benchmark, not an operational estimate*. The **chronological** split (train on earlier transactions, test on later ones) is a temporal-holdout estimate that better reflects deployment; recall at 0.5 drops from ~0.615 to ~0.49. That gap is *partly* the leakage the F1 chapter warns about and *partly* real change over time — the fraud base rate and the transaction mix both shift across the data's two days — so it should not be read as a pure leakage measurement. (Resolves the review's "random split undercuts its own leakage lesson.")

**"PR-AUC" vs AP.** `average_precision_score` computes **Average Precision (AP)**, the precision-recall summary; the script labels it AP (loosely called "PR-AUC"). It's often more informative than AUROC under strong imbalance (Saito & Rehmsmeier, 2015) — not "the honest number," since no metric is honest by itself.

## Calibration (ch. 3) → `code/calibration_analysis_wdbc.py`
Real WDBC data (bundled in scikit-learn, no download). Figures: `fig_calibration_reliability.png`, `fig_calibration_overconfidence.png`. *(Reports ECE + Brier + log loss, and runs a strictly-nested train→calibrate→test cross-check alongside the out-of-fold numbers; the test-set ECE improvement reproduces on the held-out split.)*

## Decide (ch. 5) → `code/build_decide_experiment.py`
Synthetic, fixed seed, no download. Figures: `fig_decide_thresholds.png`, `fig_decide_curve.png` (+ `decide_data.npz`), written to `handbook/figures/`.

## Reproducibility note

The **numbers** above rebuild from a clean checkout (fixed seeds throughout). The **figures** may not be pixel-identical across machines — fonts and matplotlib versions vary, so text spacing and anti-aliasing can differ — but the plotted *values* are the same. Reproduce the numbers; don't expect a byte-identical PNG.

## Not script-generated (conceptual illustrations)
`fig_auroc_ranking.png` (a hand-drawn "sorted queue" schematic), `fig_decide_confusion.png`, `fig_decide_table.png` — hand-built visuals, not data plots. Listed here so their provenance is explicit.
