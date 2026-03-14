# Network Intrusion Detection System — NSL-KDD

Binary classification of network traffic as **Normal** or **Attack** using three supervised learning algorithms evaluated on the NSL-KDD benchmark dataset.

**McMaster University — Data Mining Course Project — March 2026**

---

## Algorithms

| Model | Key property |
|---|---|
| K-Nearest Neighbours (KNN) | K=1, tuned via 5-fold CV |
| Gaussian Naive Bayes | Probabilistic baseline |
| SVM (RBF kernel) | C=10, gamma=scale, 25k stratified sample |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| KNN (K=1) | 0.7924 | 0.9743 | 0.6526 | 0.7817 | 0.8149 |
| Naive Bayes | 0.7709 | 0.9161 | 0.6578 | 0.7657 | 0.8397 |
| SVM (RBF) | 0.7932 | 0.9739 | 0.6542 | 0.7826 | **0.9283** |

**SVM is the recommended model** based on highest AUC (0.9283) and average precision (0.9378).

---

## Project Structure

```
nids-nsl-kdd/
├── data/
│   ├── raw/               ← KDDTrain.csv, KDDTest.csv (not tracked in git)
│   └── processed/         ← preprocessed CSVs (not tracked in git)
├── notebooks/
│   ├── 5_1_preprocessing.ipynb
│   ├── 5_2_modelling.ipynb
│   └── 5_3_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocess.py      ← data loading, encoding, normalisation
│   ├── train.py           ← KNN, NB, SVM training + stability analysis
│   └── evaluate.py        ← ROC, PR curves, final comparison table
├── models/                ← saved model files (not tracked in git)
├── reports/               ← all output plots and CSV table
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/nids-nsl-kdd.git
cd nids-nsl-kdd
```

**2. Create and activate virtual environment**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download from [Kaggle — NSL-KDD](https://www.kaggle.com/datasets/helenshek/nsl-kdd) and place the files at:
```
data/raw/KDDTrain.csv
data/raw/KDDTest.csv
```

**5. Launch Jupyter**
```bash
jupyter notebook
```

Run the notebooks in order:
1. `notebooks/5_1_preprocessing.ipynb`
2. `notebooks/5_2_modelling.ipynb`
3. `notebooks/5_3_evaluation.ipynb`

---

## Pipeline

```
Raw CSVs
   ↓
5_1_preprocessing   →  label encoding, MinMax scaling, binary labels
   ↓
data/processed/     →  X_train.csv, X_test.csv, y_train.csv, y_test.csv
   ↓
5_2_modelling       →  KNN (K-selection CV), Naive Bayes, SVM (grid search)
                        + stability analysis (10×5 RSKF)
                        + learning curves
   ↓
5_3_evaluation      →  ROC curves, PR curves, radar chart, final table
   ↓
reports/            →  all plots + final_comparison_table.csv
```

---

## Dataset

**NSL-KDD** is a refined version of the KDD Cup 1999 dataset that eliminates duplicate records and balances difficulty. It contains 41 features per connection organised into four groups: Basic, Content, Time-traffic, and Host-traffic.

- Training set: 125,973 instances
- Test set: 22,543 instances
- Label mapping: `normal` → 0, all attack subtypes → 1

> The test set contains attack subtypes in proportions that differ from training — this is intentional and tests generalisation to unseen attack patterns.

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## Notes

- `data/` and `models/` are excluded from git (see `.gitignore`) — download the dataset separately
- SVM is trained on a 25,000-instance stratified sample due to O(n²) training complexity
- All encoders and scalers are fitted on training data only to prevent data leakage
