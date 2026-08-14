"""
metrics_sim.py — a tiny, reusable simulator for teaching evaluation metrics.

Idea: you don't need a real model to study a metric's behavior. You need
control over the two things that drive every metric:
  - prevalence  : fraction of positives (how rare the thing you care about is)
  - separation  : how far apart the model's scores are for pos vs neg
                  (this IS discriminability; it sets AUROC directly)

Relabel "positive" as fraud / disease / defect / churn — the math is identical.
"""
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, roc_auc_score, average_precision_score)

def sigmoid(x): return 1/(1+np.exp(-x))

def simulate(n=100_000, prevalence=0.01, separation=2.0, seed=0):
    """Return (y_true, scores). Negatives ~ N(0,1), positives ~ N(separation,1),
    squashed to probability-like scores in [0,1]."""
    rng = np.random.default_rng(seed)
    n_pos = int(round(n*prevalence)); n_neg = n - n_pos
    latent = np.concatenate([rng.normal(-separation/2,1,n_neg), rng.normal(separation/2,1,n_pos)])
    y = np.concatenate([np.zeros(n_neg,int), np.ones(n_pos,int)])
    order = rng.permutation(n)
    return y[order], sigmoid(latent)[order]

def evaluate(y, scores, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    tn,fp,fn,tp = [int(v) for v in confusion_matrix(y,pred).ravel()]
    return dict(threshold=threshold, TP=tp, FP=fp, FN=fn, TN=tn,
                accuracy=accuracy_score(y,pred),
                precision=precision_score(y,pred,zero_division=0),
                recall=recall_score(y,pred,zero_division=0),
                f1=f1_score(y,pred,zero_division=0),
                AUROC=roc_auc_score(y,scores),
                PR_AUC=average_precision_score(y,scores))
