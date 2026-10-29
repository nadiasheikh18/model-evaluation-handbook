"""
sim_demo.py — watch metrics move without training a model.

The point: you don't need a real model to study a metric. You need control over
the two things that drive every metric — prevalence and separation. Relabel
"positive" as fraud / disease / defect / churn; the math is identical.

Run:  python simulator/sim_demo.py
"""
from metrics_sim import simulate, evaluate

def row(label, r):
    print(f"{label:<28} acc={r['accuracy']:.3f}  prec={r['precision']:.3f}  "
          f"recall={r['recall']:.3f}  F1={r['f1']:.3f}  AUROC={r['AUROC']:.3f}  PR-AUC={r['PR_AUC']:.3f}")

if __name__ == "__main__":
    print("Same model quality (separation=2.0), only the base rate changes:\n")
    for prev in (0.10, 0.01, 0.001):
        y, s = simulate(n=100_000, prevalence=prev, separation=2.0, seed=0)
        row(f"prevalence = {prev:>6.3%}", evaluate(y, s, threshold=0.5))

    print("\nNotice: AUROC barely moves (it tracks separation, not the base rate),")
    print("while precision and F1 fall off a cliff.")
    print("Accuracy barely moves here too -- but only because this symmetric setup makes")
    print("sensitivity ~= specificity at threshold 0.5. In general accuracy =")
    print("prevalence*sensitivity + (1-prevalence)*specificity, so it DOES drift with")
    print("prevalence (upward, toward specificity, as positives get rarer).")
    print("Either way: a metric can look 'fine' while the model is unusable on rare events.\n")

    print("Now hold prevalence at 1% and improve the model (raise separation):\n")
    for sep in (1.0, 2.0, 3.0):
        y, s = simulate(n=100_000, prevalence=0.01, separation=sep, seed=0)
        row(f"separation = {sep:>3.1f}", evaluate(y, s, threshold=0.5))
    print("\nSeparation sets AUROC directly: ~1.0 -> ~0.76, ~2.0 -> ~0.92, ~3.0 -> ~0.98.")
