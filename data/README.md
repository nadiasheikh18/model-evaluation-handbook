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

## Verify your download

The scripts expect this exact file. After downloading, confirm it matches:

| Property | Expected value |
|---|---|
| Filename | `creditcard.csv` |
| Size | `150,828,752` bytes |
| SHA-256 | `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89` |
| Rows | 284,807 transactions (+ 1 header row) |
| Positives | 492 fraud (`Class == 1`), a 0.173% base rate |
| Columns (31) | `Time`, `V1`–`V28`, `Amount`, `Class` |

```bash
# from the repo root
test "$(wc -c < data/creditcard.csv)" = 150828752 && echo "size OK"
shasum -a 256 data/creditcard.csv   # compare to the SHA-256 above (use sha256sum on Linux)
head -1 data/creditcard.csv         # header: "Time","V1",...,"V28","Amount","Class"
```

If the size or checksum differs, you have a different revision of the dataset and the chapter numbers may not reproduce exactly.

### Optional: Kaggle CLI

```bash
pip install kaggle            # then add your Kaggle API token (~/.kaggle/kaggle.json)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/ --unzip
```

## Citation & license

The dataset was collected by the **Machine Learning Group at ULB (Université Libre de Bruxelles)**. Please cite:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE Symposium Series on Computational Intelligence (SSCI), 2015.

Use of the dataset is subject to the terms on its Kaggle page.
