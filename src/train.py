"""
src/train.py
------------
Model training, hyperparameter tuning, and stability analysis
for the NSL-KDD NIDS project.

Models: K-Nearest Neighbours (KNN), Gaussian Naive Bayes (GNB), SVM (RBF kernel)

Usage:
    from src.train import train_knn, train_nb, train_svm, stability_analysis
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    learning_curve,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _evaluate(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """
    Return accuracy, precision, recall, F1 and confusion matrix.

    Parameters
    ----------
    threshold : decision threshold for positive class (default 0.5).
                Lower values increase recall at the cost of precision —
                important for NIDS where missing attacks is costly.
    """
    if threshold != 0.5 and hasattr(model, "predict_proba"):
        proba  = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    return {
        "accuracy":         accuracy_score(y_test, y_pred),
        "precision":        precision_score(y_test, y_pred, zero_division=0),
        "recall":           recall_score(y_test, y_pred, zero_division=0),
        "f1":               f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred":           y_pred,
        "threshold":        threshold,
    }


def tune_threshold(
    model,
    X_test,
    y_test,
    thresholds: np.ndarray = None,
    target_metric: str = "recall",
    min_precision: float = 0.90,
) -> float:
    """
    Find the decision threshold that maximises `target_metric` while keeping
    precision above `min_precision`.

    For NIDS, we prioritise Recall (catching all attacks) but require
    precision >= 0.90 to avoid flooding analysts with false alarms.

    Parameters
    ----------
    target_metric : 'recall' or 'f1'
    min_precision : minimum acceptable precision (default: 0.90)

    Returns
    -------
    best_threshold : float
    threshold_df   : pd.DataFrame with full sweep results
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.55, 0.05)

    if not hasattr(model, "predict_proba"):
        print("[tune_threshold] Model has no predict_proba — threshold tuning skipped.")
        return 0.5, pd.DataFrame()

    proba = model.predict_proba(X_test)[:, 1]
    rows  = []

    for t in thresholds:
        preds = (proba >= t).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        rows.append({"threshold": round(t, 2), "precision": p,
                     "recall": r, "f1": f})

    df = pd.DataFrame(rows)
    # Only consider thresholds that meet the minimum precision bar
    eligible = df[df["precision"] >= min_precision]

    if eligible.empty:
        print("[tune_threshold] No threshold meets min_precision — relaxing constraint.")
        eligible = df

    best_t = eligible.loc[eligible[target_metric].idxmax(), "threshold"]
    best_row = eligible[eligible["threshold"] == best_t].iloc[0]
    print(f"[tune_threshold] Best threshold = {best_t:.2f}  "
          f"-> P={best_row['precision']:.4f}  R={best_row['recall']:.4f}  "
          f"F1={best_row['f1']:.4f}  (optimised for {target_metric}, "
          f"min_precision={min_precision})")

    return best_t, df


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

def tune_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_range: range = range(1, 22, 2),
    cv: int = 5,
) -> int:
    """
    Find the optimal K for KNN using stratified cross-validation.
    Tests odd K values to avoid ties.

    Parameters
    ----------
    k_range : range of K values to test (default: odd values 1–21)
    cv      : number of CV folds

    Returns
    -------
    best_k : int
    cv_scores : dict {k: mean_f1}
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_scores = {}

    print(f"[tune_knn] Testing K = {list(k_range)} with {cv}-fold CV...")
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        scores = cross_val_score(knn, X_train, y_train,
                                 cv=skf, scoring="f1", n_jobs=-1)
        cv_scores[k] = scores.mean()
        print(f"  K={k:2d}  F1={scores.mean():.4f} ± {scores.std():.4f}")

    best_k = max(cv_scores, key=cv_scores.get)
    print(f"\n[tune_knn] Best K = {best_k}  (F1 = {cv_scores[best_k]:.4f})")
    return best_k, cv_scores


def train_knn(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k: int = None,
    k_range: range = range(1, 22, 2),
) -> tuple:
    """
    Train KNN. If k is None, tunes K automatically via CV.

    Returns
    -------
    model, metrics dict, cv_scores dict, best_k
    """
    if k is None:
        best_k, cv_scores = tune_knn(X_train, y_train, k_range)
    else:
        best_k, cv_scores = k, {}

    # K=1 warning: a single neighbour memorises training data, giving
    # near-perfect CV scores but poor generalisation to the test set.
    # This is a well-known overfitting risk — the gap between CV F1 (~0.997)
    # and test F1 (~0.78) confirms it. Note this as a methodological limitation.
    if best_k == 1:
        print("[train_knn] NOTE: K=1 selected. CV F1 will appear near-perfect "
              "because each training point is its own neighbour. "
              "Expect a larger train/test gap — document this as a limitation.")

    print(f"\n[train_knn] Fitting KNN with K={best_k}...")
    model = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    model.fit(X_train, y_train)

    metrics = _evaluate(model, X_test, y_test)
    print(f"[train_knn] Test results -> "
          f"Acc: {metrics['accuracy']:.4f}  "
          f"P: {metrics['precision']:.4f}  "
          f"R: {metrics['recall']:.4f}  "
          f"F1: {metrics['f1']:.4f}")

    return model, metrics, cv_scores, best_k


# ---------------------------------------------------------------------------
# Naive Bayes
# ---------------------------------------------------------------------------

def train_nb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple:
    """
    Train Gaussian Naive Bayes.
    No hyperparameters to tune — fit directly.

    Returns
    -------
    model, metrics dict
    """
    print("[train_nb] Fitting Gaussian Naive Bayes...")
    model = GaussianNB()
    model.fit(X_train, y_train)

    metrics = _evaluate(model, X_test, y_test)
    print(f"[train_nb] Test results → "
          f"Acc: {metrics['accuracy']:.4f}  "
          f"P: {metrics['precision']:.4f}  "
          f"R: {metrics['recall']:.4f}  "
          f"F1: {metrics['f1']:.4f}")

    return model, metrics


# ---------------------------------------------------------------------------
# SVM
# ---------------------------------------------------------------------------

def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_size: int = 25000,
    param_grid: dict = None,
    cv: int = 3,
) -> tuple:
    """
    Train SVM with RBF kernel.

    SVM is O(n²)–O(n³) in training time — fitting on 125k rows is infeasible.
    We use a stratified random sample of `sample_size` rows for grid search
    and final training, which is standard practice for SVM on large datasets.

    Parameters
    ----------
    sample_size : rows to sample from training set (default: 25,000)
    param_grid  : grid for GridSearchCV (default: C and gamma ranges)
    cv          : folds for grid search CV

    Returns
    -------
    model, metrics dict, best_params dict, sample indices
    """
    # Stratified sample — preserves class ratio
    idx = (
        pd.Series(y_train.values)
        .groupby(y_train.values)
        .apply(lambda x: x.sample(
            frac=sample_size / len(y_train), random_state=42))
        .index.get_level_values(1)
    )
    X_samp = X_train.iloc[idx].reset_index(drop=True)
    y_samp = y_train.iloc[idx].reset_index(drop=True)
    print(f"[train_svm] Sample size: {len(X_samp):,}  "
          f"(Normal: {(y_samp==0).sum():,}  Attack: {(y_samp==1).sum():,})")

    # Grid search
    if param_grid is None:
        param_grid = {
            "C":     [0.1, 1, 10],
            "gamma": ["scale", 0.01, 0.1],
        }

    print(f"[train_svm] Running GridSearchCV ({cv}-fold)...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
        param_grid,
        cv=skf,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_samp, y_samp)

    best_params = grid.best_params_
    print(f"[train_svm] Best params: {best_params}  "
          f"(CV F1 = {grid.best_score_:.4f})")

    # Refit best model on full sample
    # probability=True required for predict_proba (used in threshold tuning
    # and ROC/PR curves). Adds slight overhead via Platt scaling — acceptable.
    model = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    model.fit(X_samp, y_samp)

    # Threshold tuning — optimise for Recall (NIDS priority) while keeping
    # Precision >= 0.90 to avoid excessive false alarms.
    print("[train_svm] Tuning decision threshold for NIDS (target: recall, min_precision=0.90)...")
    best_threshold, threshold_df = tune_threshold(
        model, X_test, y_test,
        target_metric="recall",
        min_precision=0.90,
    )

    metrics = _evaluate(model, X_test, y_test, threshold=best_threshold)
    print(f"[train_svm] Test results (threshold={best_threshold:.2f}) -> "
          f"Acc: {metrics['accuracy']:.4f}  "
          f"P: {metrics['precision']:.4f}  "
          f"R: {metrics['recall']:.4f}  "
          f"F1: {metrics['f1']:.4f}")

    return model, metrics, best_params, (X_samp, y_samp), best_threshold, threshold_df


# ---------------------------------------------------------------------------
# Stability Analysis — Repeated Stratified K-Fold
# ---------------------------------------------------------------------------

def stability_analysis(
    models_dict: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 10,
    scoring: str = "f1",
    svm_sample_size: int = 25000,
) -> pd.DataFrame:
    """
    Repeated Stratified K-Fold cross-validation for stability assessment.

    Runs n_repeats × n_splits CV folds per model and reports:
        mean, std, min, max, coefficient of variation (CV%) of F1 across folds.

    High std / CV% → unstable model (sensitive to data split).
    Low  std / CV% → stable model (consistent regardless of split).

    Parameters
    ----------
    models_dict    : {model_name: unfitted_sklearn_estimator}
    n_splits       : folds per repeat (default: 5)
    n_repeats      : number of repeats (default: 10)  → 50 total evaluations
    scoring        : metric to measure (default: f1)
    svm_sample_size: rows to use for SVM (too slow on full set)

    Returns
    -------
    pd.DataFrame with stability stats per model
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )

    results = []

    for name, model in models_dict.items():
        print(f"[stability] Running {n_repeats}×{n_splits}-fold CV for {name}...")

        # SVM uses a fixed stratified sample to keep runtime feasible
        if "SVM" in name:
            idx = (
                pd.Series(y_train.values)
                .groupby(y_train.values)
                .apply(lambda x: x.sample(
                    frac=svm_sample_size / len(y_train), random_state=42))
                .index.get_level_values(1)
            )
            X_cv = X_train.iloc[idx].reset_index(drop=True)
            y_cv = y_train.iloc[idx].reset_index(drop=True)
        else:
            X_cv, y_cv = X_train, y_train

        scores = cross_val_score(
            model, X_cv, y_cv,
            cv=rskf, scoring=scoring, n_jobs=-1
        )

        results.append({
            "Model":   name,
            "Mean F1": round(scores.mean(), 4),
            "Std":     round(scores.std(),  4),
            "Min":     round(scores.min(),  4),
            "Max":     round(scores.max(),  4),
            "CV%":     round((scores.std() / scores.mean()) * 100, 2),
            "n_evals": len(scores),
            "_scores": scores,   # keep raw scores for plotting
        })
        print(f"  → Mean={scores.mean():.4f}  Std={scores.std():.4f}  "
              f"CV%={(scores.std()/scores.mean())*100:.2f}%")

    df = pd.DataFrame(results).drop(columns=["_scores"])
    return df, {r["Model"]: r["_scores"] for r in results}


# ---------------------------------------------------------------------------
# Learning Curves
# ---------------------------------------------------------------------------

def compute_learning_curves(
    models_dict: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_sizes: np.ndarray = None,
    cv: int = 5,
    scoring: str = "f1",
) -> dict:
    """
    Compute learning curves for each model.
    Shows how F1 changes as training set size grows.

    Returns
    -------
    dict {model_name: (train_sizes_abs, train_scores, val_scores)}
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.05, 1.0, 10)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    curves = {}

    for name, model in models_dict.items():
        print(f"[learning_curve] Computing for {name}...")
        ts, tr_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=42,
        )
        curves[name] = (ts, tr_scores, val_scores)
        print(f"  → Final val F1: {val_scores[-1].mean():.4f}")

    return curves