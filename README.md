# The Model Evaluation Handbook

**Understanding how we actually know whether an ML model can be trusted — one metric at a time, with real data, plain language, and runnable code.**

Part of **[Beyond Accuracy](https://medium.com/@nadiasheikh18)** by Nadia Sheikh. *Question assumptions. Build intuition.*

Most model-evaluation writing is either a wall of formulas or a vague blog post. This handbook is the middle path: each chapter takes one metric, builds the *intuition* from a question someone was actually forced to answer, proves it on a **real dataset**, and hands you the **code and a simulator** so you can watch the metric move yourself.

The running example is credit-card fraud (public data, obvious stakes), but every idea is portable — relabel "fraud" as disease, defect, or churn and the math is identical. The clinical mirror is called out throughout.

---

## Chapters

1. **[The Evaluation Journey](handbook/00-the-evaluation-journey.md)** — why 90% accuracy doesn't mean you should trust a model, and the families of questions evaluation actually answers.
2. **[F1 & Fraud](handbook/01-f1-and-fraud.md)** — how a model can be 99.9% accurate and catch zero fraud; the confusion matrix, precision, recall, and why F1 is the "weakest-link" harmonic mean.
3. **[AUROC](handbook/02-auroc.md)** — ranking vs. deciding, what the ROC curve really is, and why a 0.96 AUROC can still bury you in false alarms.
4. **Calibration** — *(coming)* when "97% sure" isn't.
5. **Three Metrics, Three Questions** — *(coming)* why you need F1, AUROC, and calibration together, and how a model can pass one while failing another.
6. **[A Trustworthy Probability Is Still Not a Decision](handbook/05-decide.md)** — once you trust the number, what threshold do you act on, and at what cost? Youden's index, the cost ratio `p_t = 1/(1+R)`, and decision curve analysis.

More planned: generalization.

---

## What's inside

```
handbook/     the chapters (Markdown) + figures
simulator/    metrics_sim.py — a tiny, reusable metrics simulator (two knobs: prevalence, separation)
code/         train_fraud_model.py — reproduces the F1/AUROC chapter numbers (real fraud data)
              build_decide_experiment.py — the synthetic screening model behind the Decide chapter
data/         how to fetch the public dataset (not committed here)
```

## Quickstart

```bash
pip install -r requirements.txt

# 1) Get the data (one-time — see data/README.md)
#    Download the ULB credit-card dataset from Kaggle into data/creditcard.csv

# 2) Reproduce the real numbers from the chapters
python code/train_fraud_model.py
#    → confusion matrix (91 / 16 / 57 / 85,279), precision/recall/F1,
#      AUROC ≈ 0.957, PR-AUC ≈ 0.71, and saves the ROC figure

# 3) Play with any metric without a real model — just two knobs
python simulator/sim_demo.py

# 4) Reproduce the Decide chapter (synthetic — no download needed)
python code/build_decide_experiment.py
#    → threshold table (0.50 / 0.20 / Youden 0.12 / 0.10), the cost-ratio
#      map, and the decision-curve summary; regenerates its two figures
```

The **simulator** is the fastest way to build intuition: you don't need a trained model to study a metric's behavior, only control over *prevalence* (how rare positives are) and *separation* (how well the model separates the classes — this sets AUROC directly). Make fraud rarer and watch precision and F1 fall off a cliff while accuracy and AUROC don't flinch.

## How this is written

Every chapter follows one rule: **if you can't explain it simply, you don't yet understand it.** Plain language first; the moment we'd reach for jargon is the moment we stop and explain. Formulas live in the references, not the prose — e.g., the AUROC = Wilcoxon/Mann–Whitney result is linked, not derived on the page.

## License

Dual-licensed:
- **Code** (`simulator/`, `code/`) — **MIT** ([LICENSE](LICENSE)).
- **Writing & figures** (`handbook/`) — **Creative Commons Attribution 4.0 (CC BY 4.0)** ([LICENSE-CONTENT](LICENSE-CONTENT)). Reuse freely with credit to *Nadia Sheikh — Beyond Accuracy*.

## Contributing

Corrections, clearer explanations, new real-data examples, and translations are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). If a sentence made you go "wait, why?", that's a bug; open an issue.

## Author

**Nadia Sheikh** — *Beyond Accuracy.* Written essays on Medium ([@nadiasheikh18](https://medium.com/@nadiasheikh18)); the video companion is *Neural Rhymes*. Find the brand as **@gobeyondaccuracy**.

⭐ If this helped, star the repo — it's how others find it.
