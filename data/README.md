# Data

The chapters use the **ULB credit-card fraud dataset** — 284,807 real transactions, 492 of them fraud (a 0.173% positive rate). It's the standard public benchmark for imbalanced classification.

**We do not commit the dataset here** (its license and ~150 MB size make that inappropriate). Download it yourself — one time:

## Get it from Kaggle

1. Go to the dataset page: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
2. Download `creditcard.csv`.
3. Place it in this folder as:

```
data/creditcard.csv
```

That's it — `code/train_fraud_model.py` reads from that path.

### Optional: Kaggle CLI

```bash
pip install kaggle            # then add your Kaggle API token (~/.kaggle/kaggle.json)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

## Citation & license

The dataset was collected by the **Machine Learning Group at ULB (Université Libre de Bruxelles)**. Please cite:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium Series on Computational Intelligence (SSCI), 2015.

Use of the dataset is subject to the terms on its Kaggle page.
