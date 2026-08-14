# Contributing

Thanks for helping make model evaluation clearer.

The bar for this handbook is simple: **explain it so a smart beginner gets it, without hand-waving.** If a sentence made you go "wait, why?", that's a bug — please open an issue.

## Good contributions

- **Corrections** — a wrong number, a broken repro, a misleading claim.
- **Clearer explanations** — a tighter analogy, a simpler wording, a better figure. (Keep it plain — no jargon standing in for a sentence.)
- **New real-data examples** — the same metric shown on a different public dataset/domain.
- **Reproducibility fixes** — anything that makes `code/` or `simulator/` easier to run.
- **Translations.**

## Style

- Short. One idea per section; lead with the point. Formulas go in references, not prose.
- Ground every claim in a plain-language "here's why."
- Don't commit datasets — link to the source and script the download (see `data/README.md`).

## How

1. Open an issue describing the change (especially for new chapters — let's agree on scope first).
2. Fork, make the change, keep diffs focused.
3. For code, make sure `python code/train_fraud_model.py` and `python simulator/sim_demo.py` still run.
4. Open a PR referencing the issue.

## Licensing of contributions

By contributing you agree that your code is provided under the repo's **MIT** license and your written/figure contributions under **CC BY 4.0**.
