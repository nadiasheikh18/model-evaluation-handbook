"""
metrics_sim.py — a tiny, reusable Python metric simulator for teaching evaluation.

Idea: you don't need a trained model to study a metric's behavior. You need
control over the two things that drive every metric:
  - prevalence  : fraction of positives (how rare the thing you care about is)
  - separation  : how far apart the model's scores are for pos vs neg
                  (this IS discriminability; it sets AUROC directly)

Relabel "positive" as fraud / disease / defect / churn — the math is identical.

NOTE ON THE SCORES: `simulate` returns sigmoid-squashed latent scores in [0,1].
They are **probability-like** (monotonic in the true likelihood) but are NOT
calibrated posterior probabilities — a score of 0.9 does not mean "90% of these
are positive." Ranking metrics (AUROC) only need the order; calibration metrics
(ECE) and fixed-threshold metrics (F1 at 0.5) depend on the actual values, which
is exactly what the --calibration demo makes visible.
"""
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score,
                             average_precision_score, brier_score_loss, log_loss)

def sigmoid(x): return 1/(1+np.exp(-x))

def simulate(n=100_000, prevalence=0.01, separation=2.0, seed=0):
    """Return (y_true, scores). Negatives ~ N(-sep/2,1), positives ~ N(+sep/2,1),
    squashed to probability-like scores in [0,1]. Inputs are validated."""
    if not (0 < prevalence < 1):
        raise ValueError(f"prevalence must be strictly between 0 and 1 (got {prevalence})")
    if separation < 0 or not np.isfinite(separation):
        raise ValueError(f"separation must be finite and >= 0 (got {separation})")
    if n < 2:
        raise ValueError(f"n must be >= 2 (got {n})")
    rng = np.random.default_rng(seed)
    n_pos = int(round(n * prevalence)); n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"prevalence {prevalence} gives a single-class sample at n={n}; "
                         f"raise n or prevalence")
    latent = np.concatenate([rng.normal(-separation/2, 1, n_neg),
                             rng.normal( separation/2, 1, n_pos)])
    y = np.concatenate([np.zeros(n_neg, int), np.ones(n_pos, int)])
    order = rng.permutation(n)
    return y[order], sigmoid(latent)[order]

def ece(y, scores, n_bins=10):
    """Expected Calibration Error (equal-width bins): average gap between the
    confidence a score states and the accuracy actually seen at that score."""
    y = np.asarray(y); scores = np.asarray(scores, float)
    edges = np.linspace(0, 1, n_bins + 1); e = 0.0; N = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        m = (scores >= lo) & (scores < hi if i < n_bins-1 else scores <= hi)
        if m.sum():
            e += m.sum()/N * abs(scores[m].mean() - y[m].mean())
    return float(e)

def evaluate(y, scores, threshold=0.5):
    if not (0 <= threshold <= 1):
        raise ValueError(f"threshold must be in [0,1] (got {threshold})")
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred).ravel()]
    return dict(threshold=threshold, TP=tp, FP=fp, FN=fn, TN=tn,
                accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                AUROC=roc_auc_score(y, scores),
                AP=average_precision_score(y, scores),   # Average Precision (the PR-curve summary)
                ECE=ece(y, scores),
                Brier=brier_score_loss(y, scores))
