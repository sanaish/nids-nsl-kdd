"""
src/evaluate.py
---------------
Evaluation utilities for the NSL-KDD NIDS project.

Covers:
    - Per-model metric summary
    - ROC curves + AUC
    - Precision-Recall curves + Average Precision
    - Final comparison table (metrics + stability)
    - Interpretation helpers

Usage:
    from src.evaluate import (
        metrics_summary, plot_roc, plot_pr,
        final_comparison_table, interpret_results
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)


# ---------------------------------------------------------------------------
# Metric summary
# ---------------------------------------------------------------------------

def metrics_summary(
    y_test: pd.Series,
    predictions: dict,
) -> pd.DataFrame:
    """
    Compute accuracy, precision, recall, F1 for multiple models.

    Parameters
    ----------
    y_test       : true labels
    predictions  : {model_name: y_pred array}

    Returns
    -------
    pd.DataFrame with one row per model
    """
    rows = []
    for name, y_pred in predictions.items():
        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
    return pd.DataFrame(rows).set_index("Model")


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc(
    y_test: pd.Series,
    proba_dict: dict,
    save_path: str = None,
) -> dict:
    """
    Plot ROC curves for all models on a single axes.

    Parameters
    ----------
    proba_dict  : {model_name: y_score}
                  y_score = predicted probability of class 1
                  (use model.predict_proba()[:,1] or model.decision_function())
    save_path   : if provided, save figure here

    Returns
    -------
    dict {model_name: auc_score}
    """
    colors = {"KNN": "#4C72B0", "Naive Bayes": "#DD8452", "SVM": "#55A868"}
    auc_scores = {}

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, y_score in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores[name] = round(roc_auc, 4)
        ax.plot(fpr, tpr,
                color=colors.get(name, "gray"),
                linewidth=2,
                label=f"{name}  (AUC = {roc_auc:.4f})")

    # Random classifier baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.50)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot_roc] Saved → {save_path}")
    plt.show()

    return auc_scores


# ---------------------------------------------------------------------------
# Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_pr(
    y_test: pd.Series,
    proba_dict: dict,
    save_path: str = None,
) -> dict:
    """
    Plot Precision-Recall curves for all models.

    PR curves are more informative than ROC when class distribution
    matters — they show the tradeoff between catching attacks (recall)
    and avoiding false alarms (precision).

    Parameters
    ----------
    proba_dict : {model_name: y_score}
    save_path  : optional save path

    Returns
    -------
    dict {model_name: average_precision_score}
    """
    colors = {"KNN": "#4C72B0", "Naive Bayes": "#DD8452", "SVM": "#55A868"}
    ap_scores = {}

    # Baseline = fraction of positives in test set
    baseline = y_test.mean()

    fig, ax = plt.subplots(figsize=(7, 6))

    for name, y_score in proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ap = average_precision_score(y_test, y_score)
        ap_scores[name] = round(ap, 4)
        ax.plot(recall, precision,
                color=colors.get(name, "gray"),
                linewidth=2,
                label=f"{name}  (AP = {ap:.4f})")

    ax.axhline(baseline, color="black", linestyle="--", linewidth=1,
               label=f"Random (AP ≈ {baseline:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot_pr] Saved → {save_path}")
    plt.show()

    return ap_scores


# ---------------------------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------------------------

def final_comparison_table(
    metrics_df:   pd.DataFrame,
    stability_df: pd.DataFrame,
    auc_scores:   dict,
    ap_scores:    dict,
) -> pd.DataFrame:
    """
    Merge test-set metrics, AUC, AP, and stability stats into one table.

    Parameters
    ----------
    metrics_df   : output of metrics_summary()
    stability_df : output of stability_analysis() from train.py
    auc_scores   : {model_name: auc}
    ap_scores    : {model_name: average_precision}

    Returns
    -------
    pd.DataFrame — the full comparison table
    """
    stab = stability_df.set_index("Model")[["Mean F1", "Std", "CV%"]]
    stab.columns = ["CV Mean F1", "CV Std", "CV%"]

    auc_s = pd.Series(auc_scores, name="AUC")
    ap_s  = pd.Series(ap_scores,  name="Avg Precision")

    table = metrics_df.join(auc_s).join(ap_s).join(stab)
    return table


# ---------------------------------------------------------------------------
# Per-model detailed report
# ---------------------------------------------------------------------------

def print_classification_reports(
    y_test: pd.Series,
    predictions: dict,
) -> None:
    """Print sklearn classification_report for each model."""
    for name, y_pred in predictions.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(classification_report(
            y_test, y_pred,
            target_names=["Normal (0)", "Attack (1)"]
        ))


# ---------------------------------------------------------------------------
# Radar chart — visual multi-metric comparison
# ---------------------------------------------------------------------------

def plot_radar(
    metrics_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Radar (spider) chart comparing Accuracy, Precision, Recall, F1
    across all models.
    """
    categories = ["Accuracy", "Precision", "Recall", "F1"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(polar=True))

    for (model, row), color in zip(metrics_df.iterrows(), colors):
        values = row[categories].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                color=color, label=model)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Model Comparison — Radar Chart", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_radar] Saved → {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Interpret results
# ---------------------------------------------------------------------------

def interpret_results(table: pd.DataFrame) -> None:
    """
    Print a plain-English interpretation of the final comparison table.
    Identifies best model by F1, Recall, and stability.
    """
    best_f1      = table["F1"].idxmax()
    best_recall  = table["Recall"].idxmax()
    best_stable  = table["CV%"].idxmin()
    best_auc     = table["AUC"].idxmax()

    print("\n" + "=" * 55)
    print("  INTERPRETATION")
    print("=" * 55)
    print(f"  Highest F1       : {best_f1}  ({table.loc[best_f1,'F1']:.4f})")
    print(f"  Highest Recall   : {best_recall}  ({table.loc[best_recall,'Recall']:.4f})")
    print(f"  Most Stable      : {best_stable}  (CV% = {table.loc[best_stable,'CV%']:.2f}%)")
    print(f"  Highest AUC      : {best_auc}  ({table.loc[best_auc,'AUC']:.4f})")
    print()

    # Primary recommendation: recall is priority metric for NIDS
    print(f"  NIDS priority metric is Recall (minimise missed attacks).")
    print(f"  → Recommended model: {best_recall}")

    if best_recall != best_f1:
        print(f"  Note: {best_f1} has higher F1 — consider if false alarm")
        print(f"  rate is also a concern in your deployment context.")
    print("=" * 55)