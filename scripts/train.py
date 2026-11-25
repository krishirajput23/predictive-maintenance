"""Train and compare failure classifiers, then persist the selected model.

Usage: python scripts/train.py [--data PATH] [--output-dir PATH] [--cpu-only]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data import load_dataset, prepare
from src.evaluation import (
    explain_features,
    metrics_at_threshold,
    plot_confusion_matrix,
    plot_importances,
    plot_pr_curves,
    plot_roc_curves,
    ranking_metrics,
    tune_threshold,
)
from src.models import build_models, fit_model, gpu_available


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--cpu-only", action="store_true",
                        help="disable GPU use even when an NVIDIA GPU is present")
    return parser.parse_args()


def main():
    args = parse_args()
    out = args.output_dir
    (out / "models").mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(args.data)
    X, y, devices, splits = prepare(frame, seed=args.seed)
    train_idx, val_idx, test_idx = (splits[k] for k in ("train", "val", "test"))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    use_gpu = not args.cpu_only and gpu_available()
    positives = int(y_train.sum())
    scale_pos_weight = float(np.sqrt((len(y_train) - positives) / positives))

    split_summary = {
        name: {
            "rows": int(len(idx)),
            "devices": int(frame.iloc[idx][config.DEVICE].nunique()),
            "failures": int(y.iloc[idx].sum()),
            "failure_rate": float(y.iloc[idx].mean()),
        }
        for name, idx in splits.items()
    }
    print(f"Loaded {len(frame):,} device-days from {args.data}")
    print(f"Features: {X.shape[1]} | GPU requested: {use_gpu}")
    print(pd.DataFrame(split_summary).T.to_string())

    models = build_models(scale_pos_weight, use_gpu)
    fitted_models, rows, val_probabilities, test_probabilities = {}, [], {}, {}
    print("\nFitting models (search scored by validation PR-AUC)...")
    for name, estimator in models.items():
        fitted, params, seconds = fit_model(
            name, estimator, X_train, y_train, devices.iloc[train_idx], use_gpu
        )
        fitted_models[name] = fitted
        val_probabilities[name] = fitted.predict_proba(X_val)[:, 1]
        test_probabilities[name] = fitted.predict_proba(X_test)[:, 1]
        val_rank = ranking_metrics(y_val, val_probabilities[name])
        test_rank = ranking_metrics(y_test, test_probabilities[name])
        rows.append({
            "model": name,
            "val_pr_auc": val_rank["pr_auc"],
            "val_roc_auc": val_rank["roc_auc"],
            "test_pr_auc": test_rank["pr_auc"],
            "test_roc_auc": test_rank["roc_auc"],
            "fit_seconds": seconds,
            "best_params": json.dumps(params, sort_keys=True),
        })
        print(f"  {name:<20} val PR-AUC {val_rank['pr_auc']:.4f} "
              f"| test PR-AUC {test_rank['pr_auc']:.4f} | {seconds:.1f}s")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / "model_comparison.csv", index=False)
    print("\n", comparison.drop(columns="best_params").to_string(index=False), sep="")

    best_name = comparison.loc[comparison["val_pr_auc"].idxmax(), "model"]
    best_model = fitted_models[best_name]
    threshold, val_f1 = tune_threshold(y_val, val_probabilities[best_name])
    test_scores = test_probabilities[best_name]
    test_metrics = metrics_at_threshold(y_test, test_scores, threshold)
    test_rank = ranking_metrics(y_test, test_scores)

    print(f"\nSelected model: {best_name}")
    print(f"Threshold tuned on validation for F1 = {val_f1:.3f} -> {threshold:.4f}")
    print("Test at that threshold:", json.dumps(test_metrics, indent=2))

    importances = explain_features(best_model, X_test, y_test, seed=args.seed)
    plot_pr_curves(test_probabilities, y_test, float(y_test.mean()),
                   config.FIGURE_DIR / "pr_curves.png")
    plot_roc_curves(test_probabilities, y_test, config.FIGURE_DIR / "roc_curves.png")
    plot_confusion_matrix(test_metrics, config.FIGURE_DIR / "confusion_matrix.png")
    plot_importances(importances, config.FIGURE_DIR / "feature_importance.png")

    scored = frame.iloc[test_idx][[config.DATE, config.DEVICE, config.TARGET]].copy()
    scored["failure_probability"] = test_scores
    scored["predicted_failure"] = (test_scores >= threshold).astype(int)
    scored.to_csv(out / "test_predictions.csv", index=False)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": args.seed,
        "gpu_requested": use_gpu,
        "features": list(X.columns),
        "split": split_summary,
        "baseline_pr_auc": float(y_test.mean()),
        "selected_model": best_name,
        "selected_params": json.loads(
            comparison.loc[comparison["model"] == best_name, "best_params"].iloc[0]
        ),
        "threshold": threshold,
        "validation_f1_at_threshold": val_f1,
        "test_ranking": test_rank,
        "test_at_threshold": test_metrics,
        "model_comparison": json.loads(comparison.to_json(orient="records")),
        "feature_importance": {
            key: float(value) for key, value in importances.items()
        },
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if best_name == "xgboost":
        # Train on the GPU when available, but ship a CPU-only booster so the
        # prototype can score plain NumPy/pandas input without a device mismatch.
        best_model.set_params(device="cpu", tree_method="hist")

    bundle = {
        "model": best_model,
        "threshold": threshold,
        "feature_columns": list(X.columns),
        "metadata": {
            "algorithm": best_name,
            "trained_at": report["generated_at"],
            "gpu_used": use_gpu and best_name == "xgboost",
            "train_failure_rate": float(y_train.mean()),
            "test_pr_auc": test_rank["pr_auc"],
            "test_roc_auc": test_rank["roc_auc"],
        },
    }
    joblib.dump(bundle, out / "models" / "model_bundle.joblib")
    print(f"\nArtifacts written to {out}")


if __name__ == "__main__":
    main()
