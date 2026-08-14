# The metrics simulator

A tiny, reusable tool for *feeling* how evaluation metrics behave — no trained model required.

The insight: every metric is driven by just two things.

- **prevalence** — how rare the positives are (fraud / disease / defect / churn).
- **separation** — how far apart the model's scores are for positives vs. negatives. This *is* discriminability, and it sets AUROC directly (≈1.0 → AUROC ~0.76, ≈2.0 → ~0.92, ≈3.0 → ~0.98).

Turn those two knobs and every metric follows. That's enough to build real intuition.

## Use it

```bash
pip install -r ../requirements.txt
python sim_demo.py
```

## API

```python
from metrics_sim import simulate, evaluate

# generate labelled scores: negatives ~ N(-sep/2,1), positives ~ N(+sep/2,1), squashed to [0,1]
y, scores = simulate(n=100_000, prevalence=0.01, separation=2.0, seed=0)

# every common metric at a chosen threshold
m = evaluate(y, scores, threshold=0.5)
# -> dict: TP, FP, FN, TN, accuracy, precision, recall, f1, AUROC, PR_AUC
```

## The lesson it teaches fastest

Hold `separation` fixed and lower `prevalence`: **accuracy and AUROC barely move while precision and F1 collapse.** Same model — only the base rate changed. That's the trap the F1 and AUROC chapters are about, and here you can watch it happen in one loop.
