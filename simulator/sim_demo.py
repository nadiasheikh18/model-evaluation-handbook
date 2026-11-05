"""
sim_demo.py — watch metrics move without training a model.

You don't need a real model to study a metric. You need control over the two
things that drive every metric — prevalence and separation. Relabel "positive"
as fraud / disease / defect / churn; the math is identical.

Usage:
  python simulator/sim_demo.py                          # the guided sweeps (default)
  python simulator/sim_demo.py --prevalence 0.01 --separation 2.0 --threshold 0.5
  python simulator/sim_demo.py --calibration            # ranking vs. calibration demo
  (also: --seed, --n)

Note: the scores are probability-like, not calibrated posteriors (see metrics_sim.py).
"""
import argparse
import numpy as np
from metrics_sim import simulate, evaluate, ece

def row(label, r):
    print(f"{label:<30} acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
          f"recall={r['recall']:.3f}  F1={r['f1']:.3f}  AUROC={r['AUROC']:.3f}  AP={r['AP']:.3f}")

def guided(n, seed):
    print("Same model quality (separation=2.0), only the base rate changes:\n")
    for prev in (0.10, 0.01, 0.001):
        y, s = simulate(n=n, prevalence=prev, separation=2.0, seed=seed)
        row(f"prevalence = {prev:>6.3%}", evaluate(y, s, threshold=0.5))
    print("\nNotice: AUROC barely moves (it tracks separation, not the base rate),")
    print("while precision and F1 fall off a cliff. Accuracy barely moves here too --")
    print("but only because this symmetric setup makes sensitivity ~= specificity at 0.5.")
    print("In general accuracy = prevalence*sensitivity + (1-prevalence)*specificity, so it")
    print("DOES drift with prevalence (toward specificity as positives get rarer).")
    print("A metric can look 'fine' while the model is unusable on rare events.\n")
    print("Now hold prevalence at 1% and improve the model (raise separation):\n")
    for sep in (1.0, 2.0, 3.0):
        y, s = simulate(n=n, prevalence=0.01, separation=sep, seed=seed)
        row(f"separation = {sep:>3.1f}", evaluate(y, s, threshold=0.5))
    print("\nSeparation sets AUROC directly: ~1.0 -> ~0.76, ~2.0 -> ~0.92, ~3.0 -> ~0.98.")
    print("(Small wobble between runs is Monte-Carlo noise from resampling, not a real effect.)")

def single(prev, sep, thr, n, seed):
    y, s = simulate(n=n, prevalence=prev, separation=sep, seed=seed)
    r = evaluate(y, s, threshold=thr)
    print(f"prevalence={prev:.3%}  separation={sep}  threshold={thr}  (n={n:,}, seed={seed})\n")
    print(f"  confusion @ {thr}:  TP={r['TP']}  FP={r['FP']}  FN={r['FN']}  TN={r['TN']:,}")
    print(f"  accuracy {r['accuracy']:.3f} | precision {r['precision']:.3f} | recall {r['recall']:.3f} | F1 {r['f1']:.3f}")
    print(f"  AUROC {r['AUROC']:.3f} | AP {r['AP']:.3f} | ECE {r['ECE']:.3f} | Brier {r['Brier']:.3f}")

def calibration(prev, sep, thr, n, seed):
    """A monotonic remap preserves ranking (AUROC) but changes calibration (ECE)
    and fixed-threshold metrics (F1 at a cutoff)."""
    y, s = simulate(n=n, prevalence=prev, separation=sep, seed=seed)
    transforms = [("raw scores  s",                 s),
                  ("sharpened   s**3",               s**3),
                  ("flattened   sqrt(s)",            np.sqrt(s))]
    print("Monotonic (rank-preserving) re-mappings of the SAME scores:")
    print(f"(prevalence={prev:.1%}, separation={sep}, threshold={thr}, n={n:,})\n")
    print(f"  {'transform':<20} {'AUROC':>7} {'ECE':>7} {'Brier':>7} {'F1@thr':>8} {'precision':>10} {'recall':>8}")
    for name, ss in transforms:
        r = evaluate(y, ss, threshold=thr)
        print(f"  {name:<20} {r['AUROC']:>7.3f} {r['ECE']:>7.3f} {r['Brier']:>7.3f} "
              f"{r['f1']:>8.3f} {r['precision']:>10.3f} {r['recall']:>8.3f}")
    print("\nAUROC is IDENTICAL across all three — a monotonic remap can't change the ranking.")
    print("But ECE, Brier, and the F1 at a fixed threshold all move: a model can rank well —")
    print("all three remaps here rank IDENTICALLY — and still be badly calibrated. Ranking quality")
    print("and probability quality are different")
    print("questions, which is why you measure both.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Teaching simulator for evaluation metrics.")
    ap.add_argument("--prevalence", type=float, help="fraction of positives, strictly in (0,1)")
    ap.add_argument("--separation", type=float, help="class separation (sets AUROC); >= 0")
    ap.add_argument("--threshold", type=float, default=0.5, help="decision threshold in [0,1]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--calibration", action="store_true",
                    help="show that a monotonic remap preserves AUROC but changes calibration/F1")
    a = ap.parse_args()
    try:
        if a.calibration:
            calibration(a.prevalence if a.prevalence is not None else 0.10,
                        a.separation if a.separation is not None else 2.0,
                        a.threshold, a.n, a.seed)
        elif a.prevalence is not None or a.separation is not None:
            single(a.prevalence if a.prevalence is not None else 0.01,
                   a.separation if a.separation is not None else 2.0,
                   a.threshold, a.n, a.seed)
        else:
            guided(a.n, a.seed)
    except ValueError as e:
        raise SystemExit(f"input error: {e}")
