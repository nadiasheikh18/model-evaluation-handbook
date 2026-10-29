# Three Metrics, Three Questions: F1, AUROC, and Calibration

*One metric can't tell you whether a model deserves your trust. F1, AUROC, and calibration reveal three different strengths — and three different ways a model can fail.*

A model lands on your desk with a row of numbers that look like a green light: **accuracy 91%, AUROC 0.96, F1 0.89.** Read those and you'd ship it tomorrow. But suppose its calibration is poor — the failure the [last chapter](https://medium.com/towards-artificial-intelligence/does-your-ai-know-when-it-might-be-wrong-1c32f1e1fffc) was about. That model separates the classes beautifully, makes the right call at its threshold, and still, when it reports "99% sure," that number can't be believed.

That's the trap this chapter is about. We've met F1, AUROC, and calibration one at a time. Now we put them side by side — because each answers a *different* question, and a model can score brilliantly on one while failing on another. The whole point: **these three metrics examine different, interconnected properties of a model.** A strong result on one does not guarantee a strong result on the others.

- **F1 → Can it predict?** When the model commits to a positive call, how well does it balance catching real cases (recall) and avoiding false alarms (precision)?
- **AUROC → Can it rank?** Can it separate the classes and order cases by risk?
- **Calibration → Can it be trusted?** When it says 82%, is that a real 82%?

![Three questions, one at a time: PREDICT (F1) — does the positive call balance recall and precision? RANK (AUROC) — does it order cases by risk? TRUST (calibration) — is the stated confidence honest? Three different questions; a model can pass one and fail another.](figures/fig_three_metrics_motif.png)

Keep the three verbs — *predict, rank, trust* — in mind. They are not interchangeable, and confusing them is how a model that looks good on paper fails where it actually gets used. (The running example here is clinical, because the stakes are easy to feel; relabel "sick" as "fraud," "defective," or "churn" and every line holds.)

## F1 — *"Can it predict?"*

**What it answers.** F1 asks how well the model, at a chosen threshold, balances *finding* the positive cases (recall) against making *reliable* positive calls (precision). It's the harmonic mean of the two, `2PR/(P+R)` — it weights precision and recall equally, which is not the same as weighing a missed case and a false alarm by their real-world cost. It can be more informative than accuracy when the positive class is rare, because it focuses on precision and recall rather than rewarding correct negatives: a model that calls all 1,000 patients healthy when only 10 are sick scores 99% accuracy and an F1 near zero. F1 refuses to reward that. (It isn't the only choice under imbalance — sometimes recall at a clinically acceptable false-positive rate is the more useful target.)

**Where it's the right tool.** When the model drives a concrete, fixed action and the positive is rare — a screening flag, a fraud alert. You've chosen an operating threshold, positives are uncommon, and you need one number that punishes both failure modes at once.

**What it can't tell you.** F1 works on hard labels at *one* threshold, so it's blind to everything the probabilities carry — it can't tell you whether a different cutoff would serve you better, and it can't tell you whether the confidence is honest. It also shifts with prevalence, so an F1 from one setting doesn't transfer to another. And it hides *which* of precision or recall is the weak one, so report those two beside it.

## AUROC — *"Can it rank?"*

**What it answers.** AUROC steps back from any single threshold and asks whether the model can *order* cases correctly: does a randomly chosen positive tend to outscore a randomly chosen negative? That plain reading is also its definition — AUROC equals the probability that a random positive outscores a random negative, counting a tie as half ([Hanley & McNeil, 1982](https://pubs.rsna.org/doi/abs/10.1148/radiology.143.1.7063747)). It summarizes separating power across every cutoff into one number, which makes it the natural metric for comparing models and for any task where a priority order — not a fixed yes/no — is the point.

**Where it's the right tool.** When the output is a ranked queue rather than a decision: a worklist that puts the most suspicious cases at the top. No threshold is fixed; the value is entirely in the ordering. It's also the right lens for choosing between two models before you've committed to an operating point.

**What it can't tell you.** AUROC won't choose your threshold — every point on the ROC curve *is* a threshold, and the number averages over all of them. It stays a valid measure of ranking even under heavy imbalance, but it can remain high while precision is poor at the prevalence and operating threshold that actually matter in deployment; under strong imbalance a precision–recall view often reveals practical limits the ROC curve hides ([Saito & Rehmsmeier, 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)). And it says nothing about whether the probabilities are trustworthy: a model can rank every case perfectly (AUROC 1.0) while every probability it emits is miscalibrated — ranking cares about *order*, not *value*.

## Calibration — *"Can it be trusted?"*

**What it answers.** Calibration asks whether predicted probabilities match observed frequencies: if a model assigns a 90% probability of disease to a group of patients, about 90% of that group should have it. (That's calibration of the *event probability*; the calibration chapter audits the related-but-distinct *top-label confidence* — worth saying which one a reported number refers to.) Reliability diagrams let you inspect this relationship; **Expected Calibration Error (ECE)** is *one* summary of the deviations — useful, but sensitive to binning and sample size, and a single aggregate score can hide real problems. This is the axis that matters when the probability itself — not just the class — drives what happens next.

**Where it's the right tool.** Whenever a graded decision hangs on the number: a 51% warrants a follow-up, a 74% a referral, a 99% a conversation about treatment. The predicted *class* can be identical in all three; only the confidence separates the actions. If that 99% is really an overconfident 90%, every action keyed to it inherits the error.

**What it can't tell you.** Calibration is silent about whether the model can tell the classes apart at all. A model that ignores every input and outputs the base rate — 8% for everyone — can be almost perfectly calibrated yet unable to discriminate between individual cases (AUROC 0.5). Calibration is a property of *honesty*, not *skill*; always read it next to a discrimination metric. And a low *overall* ECE can still hide a confidently-wrong subgroup.

![Each metric's blind spot. F1 is blind to the probabilities and to any other threshold; AUROC won't pick a threshold and can hide poor precision under imbalance; calibration ignores discrimination, so a useless base-rate model can still look well-calibrated.](figures/fig_three_metrics_blindspots.png)

## Seeing them side by side

| | **F1** | **AUROC** | **Calibration** |
|---|---|---|---|
| **Main question** | Are positive calls both found and reliable? | Are cases ranked correctly? | Do predicted probabilities match reality? |
| **Operates on** | Hard labels, at one threshold | Continuous scores (ranking) | Predicted probabilities |
| **Threshold-dependent** | Yes | No fixed threshold | Not inherently |
| **Prevalence-sensitive** | Yes | Not directly, under fixed score distributions | Can shift under a prevalence change |
| **Main blind spot** | Other thresholds, and the cost of each error | Precision and utility at the operating point | Discrimination and decision value |

## Why you need all three: one strength can't guarantee another

The reason no single metric suffices is that a strong result on one does not carry over to the others. A clean way to see it: apply a strictly increasing transformation to a model's scores and the *ranking* is untouched, so AUROC is identical — yet the calibration, and the F1 at a fixed numerical threshold, can both move. Same order, different probabilities, different thresholded decisions. The metrics are measuring genuinely different things.

Go back to the green-light row — accuracy 91%, AUROC 0.96, F1 0.89 — and add the number nobody printed: a poor ECE. That model ranks beautifully and makes the right call at its threshold, and still can't be trusted when it says "99%." Ranking and thresholded accuracy were never going to catch that; only calibration does.

The failure runs in every direction:

- **Great ranking, bad calibration.** Modern neural networks are notorious for this — excellent AUROC, systematically overconfident probabilities ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)). Fine for a worklist, dangerous for a displayed risk score.
- **Great calibration, no discrimination.** The base-rate predictor: honest probabilities, zero ability to tell cases apart.
- **Great F1 at one threshold, useless elsewhere.** F1 certifies a single operating point and says nothing about whether the model degrades gracefully if you move the cutoff — which AUROC would show you.

![The three qualities can diverge. Great ranking with bad calibration: orders cases perfectly but overstates how sure it is. Great calibration with no discrimination: honest probabilities but can't tell cases apart. Great F1 at one threshold: nails one cutoff, says nothing about any other.](figures/fig_three_metrics_whyall.png)

This isn't a new worry, and it isn't only a deep-learning one: the standard framework for evaluating clinical prediction models has long insisted on reporting discrimination *and* calibration together, alongside a measure of clinical usefulness ([Steyerberg et al., 2010](https://pubmed.ncbi.nlm.nih.gov/20010215/)).

## Which metric when

![Which metric when. A fixed yes/no action on a rare positive: lead with F1, and report precision and recall beside it. Stratifying risk, building a worklist, or comparing models: lead with AUROC, and check precision at your real prevalence. The probability itself drives a graded decision: calibration, read alongside a discrimination metric.](figures/fig_three_metrics_whichwhen.png)

- The model commits to a **yes/no action** and the positive is **rare**: lead with **F1**, and always show precision and recall beside it.
- You're **stratifying risk, building a worklist, or comparing models**: lead with **AUROC** — but check precision under your real prevalence first.
- The **probability itself is the product** — shown to a person, combined with other risks, or driving a graded decision: **calibration** is non-negotiable, read alongside a discrimination metric.

## Watch them dissociate for yourself

The fastest way to feel why one number isn't enough is to make the metrics move while you hold the model fixed. That's what the `simulator/` in this repo is for — two knobs, *prevalence* and *separation*. Lower the prevalence while holding the class-conditional score distributions fixed, and AUROC stays about the same (it tracks the model's separating power, not the base rate) while precision and F1 fall off a cliff:

```bash
python simulator/sim_demo.py
```

In this symmetric demo accuracy barely moves either — but only because sensitivity and specificity happen to be nearly equal at this threshold. In general accuracy is `prevalence × sensitivity + (1 − prevalence) × specificity`, so it drifts as the base rate changes — usually *upward* as positives get rarer, toward the specificity — which is its own reason not to lean on it. The lesson holds: these metrics don't rise and fall together, so reporting one and hoping is not evaluation.

## The next question

F1, AUROC, and calibration all grade a model on the data in front of it. But a trustworthy probability is still not a decision: once you trust the number, what threshold do you actually act on — and at what cost? That's the next chapter.

---

**Sources & further reading**
- AUROC = the probability a random positive outranks a random negative: [Hanley & McNeil, 1982 (Radiology)](https://pubs.rsna.org/doi/abs/10.1148/radiology.143.1.7063747)
- Why precision–recall beats ROC under imbalance: [Saito & Rehmsmeier, 2015 (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
- Modern networks rank well but are overconfident: [Guo et al., 2017](https://arxiv.org/abs/1706.04599)
- Report discrimination, calibration, and clinical usefulness together: [Steyerberg et al., 2010 (Epidemiology)](https://pubmed.ncbi.nlm.nih.gov/20010215/)
- Calibration as the neglected axis: [Van Calster et al., 2019 (BMC Medicine)](https://link.springer.com/article/10.1186/s12916-019-1466-7)

*By Nadia Sheikh · **Beyond Accuracy** — Question assumptions. Build intuition.*
