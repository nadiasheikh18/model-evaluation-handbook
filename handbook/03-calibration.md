# Does Your AI Know When It Might Be Wrong?

*A model can be right most of the time and still be systematically wrong about how sure it is. Calibration is the property that catches that — a separate axis from accuracy, and one worth checking alongside it.*

The last three chapters built up a model's skills. Can it make the call (**F1**)? Can it rank the risky cases near the top ([**AUROC**](https://medium.com/towards-artificial-intelligence/a-model-with-0-96-auroc-can-still-bury-you-in-false-alarms-028d05517553))? Every one of those trusted the model's scores at face value. When the model said "0.97," we used that number to threshold and to rank — but we never asked the obvious thing: *does 0.97 mean anything?* Of all the cases a model labels "97% likely," does the event really happen about 97% of the time — or does the model just sound sure?

That question has a name — **calibration** — and here's the result that made me start checking it early.

> **Experiment at a glance.** On real, public data (the Wisconsin breast-cancer set, 569 patients, which ships with scikit-learn), I train three ordinary models and grade them not only on accuracy but on whether their stated confidence can be trusted. Everything is out-of-fold, and the recalibration test below is evaluated on a held-out split. Run it yourself: `python code/calibration_analysis_wdbc.py` — it prints every number in this chapter and regenerates both figures. Three evidence levels appear here and I keep them labelled: what this **real-data experiment measures**, what a **synthetic study illustrates**, and what **published research establishes**. None of this, on its own, certifies a model as safe to deploy.

## The most accurate model was the least honest about its confidence

Train a logistic regression, a small neural network (an MLP), and a naive-Bayes classifier on the same data, and the most accurate model is the MLP — 89%, just ahead of logistic regression's 87%. But look at what happens *inside its most confident predictions*:

![Two panels on real breast-cancer data (out-of-fold). Left: overall accuracy — MLP 89%, logistic regression 87%, naive Bayes 79%. Right, leakage-free: within each model's own predictions made at over 80% confidence, the confidence it states versus the accuracy it actually delivers there. Logistic regression states 94% and delivers 94%. The MLP states 99% but delivers only 91%. Naive Bayes states 95% and delivers 86%.](figures/fig_calibration_overconfidence.png)

Within the predictions it makes at more than 80% confidence, the MLP *states* an average of **99%** and *delivers* **91%**. Logistic regression, in the same high-confidence region, states 94% and delivers 94% — its sureness is earned. Same task, same data, measured on held-out folds. One model's confidence matches its accuracy; the other's doesn't.

Accuracy alone would never have shown you this. That gap — between how sure a model *sounds* and how often it's actually right — is calibration, and it is a different axis from being correct.

## First, a distinction that matters: calibration is not "knowing when you're wrong"

It's tempting to read the above as "logistic regression knows when it might be wrong, and the MLP doesn't." That's not quite what calibration measures, and the difference is important enough to state plainly.

**Calibration** asks whether predicted probabilities agree with observed frequencies *across groups of predictions*: of all the cases labelled 90%, do about 90% turn out positive? **Error detection** — assigning lower confidence to the specific cases you'll get wrong — is a related but distinct property. A *perfectly* calibrated model can still make highly confident mistakes: among 100 predictions made at 95% confidence, five errors are exactly what perfect calibration permits. Calibration promises the *rate* is honest; it does not promise the model flags its individual mistakes in advance.

So the honest reading of the chart above is narrower and more defensible: *the MLP's stated confidence is a worse match for its observed accuracy, and a larger share of its mistakes arrive dressed as near-certainty.* Those are related, but they are not the same thing — and keeping them separate is exactly the kind of care this series is about.

## What calibration is — and four things it isn't

The cleanest definition comes from the most calibrated people on earth: weather forecasters. When a forecaster says "70% chance of rain," it should rain on about 70% of the days they say it. A model is well-calibrated when the same holds: of all the times it says "90%," it's right about 90% of the time.

Four traps:

- **It is not accuracy.** A model can be accurate and badly miscalibrated (the MLP above), or perfectly calibrated but **unable to distinguish individual risk** — predict the base rate for everyone and you're always honest and never useful. Two different axes; you want both.
- **It is not whether the model is *right*.** It's whether the model's *sureness* is trustworthy. A calibrated model that says "60%" and turns out wrong hasn't failed — it told you not to lean on it.
- **A low error score does not mean "safe."** As we'll see, the standard one-number score can look healthy while a pocket of confident mistakes hides inside it.
- **Recalibrating does not always help.** Post-hoc fixes can leave calibration worse, and their benefit has to be measured, not assumed.

> Accuracy asks *did the model get it right?* Calibration asks *was it honest about how sure it was?* Most evaluations only answer the first.

## Where the confidence number comes from

Before asking whether a confidence is honest, it helps to know what it is — because different classifiers produce it in different ways. A neural network transforms its raw output scores (the *logits*) through a step called the **softmax**, which squashes them so they land between 0 and 1 and sum to 1 across the options. Binary **logistic regression** commonly uses a **sigmoid**. **Naive Bayes** derives its probabilities from **Bayes' rule** and assumptions about how the data is distributed. These numbers can all *look* like probabilities without necessarily matching observed frequencies — and calibration is precisely the test of whether they do.

One more distinction worth carrying into any clinical reading: the number we've been auditing is **top-label confidence** — how sure the model is in whichever class it picked. For a risk decision, the quantity that usually matters more directly is the calibrated **probability of the outcome itself** (say, malignancy), not just the confidence on the winning label. They're related but not identical, and it's worth being clear about which one a reported "calibration" number refers to.

## How we measure it: bins, the reliability diagram, and ECE

You can't judge one prediction's calibration — a single guess is just right or wrong. So you group predictions by their stated confidence into **bins**, and in each bin ask: what was the *average confidence* here, and what was the *actual accuracy* here? Plot those against each other and you get a **reliability diagram**. Perfect calibration is the 45° line. Points *below* it mean overconfidence; points *above* mean underconfidence.

![Reliability diagram on the real breast-cancer data, out-of-fold. Logistic regression (teal) tracks the diagonal — where it says 80%, it's right about 80% of the time. The MLP (red) makes almost all its predictions at high confidence, and across that range its points sit below the line: it is right less often than it claims.](figures/fig_calibration_reliability.png)

Squeeze the whole diagram into one number and you get **Expected Calibration Error (ECE)**: the average gap between confidence and accuracy across the bins, weighted by how many predictions each bin holds.

```
ECE  =  Σ_bins  (n_bin / N) · | accuracy_bin − confidence_bin |
```

Here `n_bin` is how many predictions fall in a bin, `N` the total, and the bars are an absolute gap. Lower is better. Logistic regression scores about **0.03**, the MLP about **0.09**. But ECE is an average, and averages bury their worst cases.

## Why one number is never enough — report three

Take the headline comparison — "80% of the MLP's errors came while it was over 80% confident" — and notice the catch: that number is sensitive to *how often* a model makes high-confidence predictions at all. The MLP calls **94% of all its cases** at over 80% confidence, so of course most of its mistakes land there. A model that's confident about everything will make most of its errors "confidently" even if it's reasonably calibrated. The share-of-errors number alone can mislead.

The fix is to report three quantities together, because each answers a different question:

| Model | ECE | Share of errors made at >80% confidence | Error rate *among* >80%-confidence predictions |
|---|---|---|---|
| Logistic regression | 0.03 | 33% | 6% |
| MLP | 0.09 | 76% | 9% |
| Naive Bayes | 0.08 | 50% | 14% |

ECE says how closely confidence matches accuracy overall. The **share of errors at high confidence** says where a model's mistakes are concentrated — useful, but inflated when a model is confident about everything. The **error rate among high-confidence predictions** is the one that cuts through: of the cases where the model claimed to be very sure, how often was it wrong? When a model states ~99% and is wrong 9% of the time in that region, the gap is the overconfidence, and it's immune to the concentration artifact. In a high-stakes setting that error-rate-when-sure is another number worth examining closely — though an actual deployment decision also needs error consequences, subgroup checks, and external validation that no single table provides.

(There's a third hiding place specific to multi-class problems, where a model can quietly shift probability away from a rare-but-treatable class toward "nothing's wrong." I worked that case through on a *synthetic* four-stage staging model, where the worst-calibrated stage was the one that overlapped its neighbours and the rarest stage was actually the best-calibrated. That study **illustrates** that calibration failure can track class *overlap*, not just rarity — but it's only one example: in general both overlap and class imbalance can matter, depending on the model, the training procedure, and the evaluation distribution. The walkthrough is in the [full essay on Medium](https://medium.com/towards-artificial-intelligence/does-your-ai-know-when-it-might-be-wrong-1c32f1e1fffc).)

## Fixing it — carefully, and measuring honestly

If a model's confidence is dishonest, you don't have to retrain it. You can leave the model alone and re-map its confidence numbers afterward — **post-hoc recalibration**: learn a small correction on held-out data, then apply it. Three common tools, flexible to gentle:

- **Platt scaling** fits one smooth S-shaped curve from scores to probabilities. Few knobs; it can straighten a *consistent* overconfidence but nothing more complex.
- **Isotonic regression** fits a free-form staircase (only ever upward). Powerful with lots of data; on small data it overfits.
- **Temperature scaling** divides the logits by a single positive number **T** before the softmax. Because *one* uniform positive temperature is applied to *all* the class logits, their order is preserved — so the predicted class, and therefore accuracy, stays the same (apart from numerical or tie-handling edge cases). It's a simple, useful baseline, especially for neural-network logits — though it is not automatically the safest choice for every model or dataset.

Here's the honest part — and note how it's measured. I fit a temperature on the MLP using one half of the out-of-fold predictions, then reported the result on the *other, held-out half*, so the "after" number isn't scored on the data used to pick it. On that independent half, ECE improved from **0.083 to 0.051** (T ≈ 3.3), with accuracy unchanged — a real but partial fix, not a cure.

And recalibration can backfire. On a separate model I'd trained with class-weighting (to protect a rare class), Platt scaling and isotonic regression *increased* measured ECE on the evaluation data. That result is consistent with the methods being poorly matched to the available scores and sample size — the scores were deliberately skewed to protect the rare class, and a flexible method can re-learn that skew — but a single run does not establish one cause. The durable lesson is procedural: **recalibration is a hypothesis to test, not a guarantee.** Measure ECE before and after, on data you didn't use to fit the correction, prefer the gentle option first, and be willing to report "we recalibrated, and it didn't help."

## This is as much an AI problem as a legacy one

Language models introduce a related challenge: their answers can sound authoritative even when they're wrong. But linguistic confidence and probabilistic calibration are not identical — fluent prose is not a numerical probability. To measure calibration you need a defined confidence signal and a way to compare it against observed correctness. Research on GPT-4 **establishes** that training stages change probabilistic calibration: OpenAI's [technical report](https://arxiv.org/abs/2303.08774) found the *pretrained* model was well-calibrated and that **RLHF** (the human-feedback fine-tuning that makes a model helpful and agreeable) made its calibration worse. That makes calibration a live question for LLM-based systems and for AI evaluators — when we reach AI grading AI (LLM-as-a-judge), we'll have to ask it of the judge too: not just what it decides, but whether its stated confidence can be trusted.

## Limitations

A few caveats worth stating plainly. The breast-cancer dataset is small — 569 patients — so each held-out fold carries only about 57, and the high-confidence bins hold few cases each. That makes every ECE here an *estimate*: real, but noisy, and sensitive to how many bins you use and how the evaluation is split. The numbers are meant to make calibration legible, not to certify any of these models as ready for clinical use — that would take far more data, prospective validation, subgroup analysis, and a hard accounting of what each kind of error costs. Read every figure in this chapter as an illustration of the *measurement*, not a verdict on the *model*.

## The next question

We've now asked three things of a model: can it predict (F1), can it rank (AUROC), and does its stated confidence match reality (calibration)? Each is a different axis, and a model can ace one while failing another. The next chapter puts all three side by side.

## The one line to remember

A model can be accurate without being calibrated. It can be calibrated while still making confident mistakes. And it can post a reassuring calibration score while hiding real weakness in a particular subgroup or probability range. **Calibration is not the end of evaluation — it's another question we've learned to ask.**

## Reproduce it

Every number and both figures come from one script in this repo, on real public data (no download — the dataset ships with scikit-learn):

```bash
pip install -r requirements.txt
python code/calibration_analysis_wdbc.py
#  → trains LogReg / MLP / naive-Bayes on the Wisconsin breast-cancer set and prints,
#    for each: accuracy, ECE, the share of errors at >80% confidence, and the error
#    rate among >80%-confidence predictions; then fits temperature scaling on one half
#    of the out-of-fold predictions and reports ECE on the held-out half; saves both figures.
```

The controls that make the numbers honest: probabilities are out-of-fold (`cross_val_predict` over a fixed-seed `StratifiedKFold`, with the scaler and model inside a `Pipeline` so preprocessing is fit per training fold, never on the row being scored); the temperature is fit and evaluated on disjoint halves; and the seed, bin count, and confidence threshold are fixed constants at the top of the file, so the printed values are the values quoted here.

---

**Sources & further reading**
- ECE, temperature scaling, and why modern networks are overconfident: [Guo, Pleiss, Sun & Weinberger, 2017 — *On Calibration of Modern Neural Networks*](https://arxiv.org/abs/1706.04599)
- Pretrained calibration vs. post-RLHF change: [OpenAI, 2023 — *GPT-4 Technical Report*](https://arxiv.org/abs/2303.08774)
- The overconfidence-from-RLHF effect, studied directly: [Leng et al., 2024 — *Taming Overconfidence in LLMs*](https://arxiv.org/abs/2410.09724)
- Why calibration is the quietly-neglected axis in clinical models: [Van Calster et al., 2019 — *Calibration: the Achilles Heel of Predictive Analytics* (BMC Medicine)](https://link.springer.com/article/10.1186/s12916-019-1466-7)
- The deeper multi-class walkthrough (synthetic staging model): [*Does Your AI Know When It Might Be Wrong?* on Medium](https://medium.com/towards-artificial-intelligence/does-your-ai-know-when-it-might-be-wrong-1c32f1e1fffc)

*By Nadia Sheikh · **Beyond Accuracy** — Question assumptions. Build intuition.*
