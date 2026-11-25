"""Imbalance-aware metrics, threshold selection and diagnostic plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def ranking_metrics(y_true, probability):
    return {
        "pr_auc": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
    }


def tune_threshold(y_true, probability):
    """Pick the threshold with the highest F1 on the validation split."""
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1])
    f1 = np.nan_to_num(f1)
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def metrics_at_threshold(y_true, probability, threshold):
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def explain_features(model, X, y, seed=42):
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    return (
        pd.Series(result.importances_mean, index=X.columns)
        .sort_values(ascending=False)
    )


def plot_pr_curves(probabilities, y_true, baseline, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, probability)
        ax.plot(recall, precision, lw=1.6, label=name)
    ax.axhline(baseline, color="grey", ls="--", lw=1, label=f"no-skill ({baseline:.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall curves (test)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc_curves(probabilities, y_true, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_true, probability)
        ax.plot(fpr, tpr, lw=1.6, label=name)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves (test)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(metrics, path):
    matrix = np.array([[metrics["tn"], metrics["fp"]],
                       [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["predicted OK", "predicted fail"],
        yticklabels=["actual OK", "actual fail"],
    )
    ax.set_title(f"Confusion matrix at threshold {metrics['threshold']:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_importances(importances, path, top_n=15):
    selected = importances.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(selected.index, selected.values, color="#3b6ea5")
    ax.set(xlabel="Mean drop in PR-AUC when permuted",
           title="Feature importance (test split)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
