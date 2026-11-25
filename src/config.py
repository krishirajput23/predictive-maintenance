"""Paths, column names and experiment settings shared by the training and inference code."""

from pathlib import Path

import numpy as np
import pandas as pd

# The day-index feature is relative to a fixed origin so that new input files
# produce the same feature values as the training data.
DATE_ORIGIN = pd.Timestamp("2015-01-01")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "predictive_maintenance_dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "models" / "model_bundle.joblib"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
COMPARISON_PATH = OUTPUT_DIR / "model_comparison.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"

TARGET = "failure"
DEVICE = "device"
DATE = "date"
METRICS = [f"metric{i}" for i in range(1, 10)]
# metric8 is an exact duplicate of metric7 in the supplied file and adds no signal.
DROPPED_METRICS = ["metric8"]

RANDOM_STATE = 42
TEST_DEVICE_FRACTION = 0.30  # of all devices, split evenly into validation and test
CV_SPLITS = 3
SEARCH_ITERATIONS = 4

# Which hyperparameters are searched for the tree ensembles. The remaining
# settings are fixed in models.py and documented in README.md.
SEARCH_SPACE = {
    "random_forest": {
        "max_depth": [None, 16],
        "min_samples_leaf": [1, 8],
        "max_features": ["sqrt", 0.5],
    },
    "gradient_boosting": {
        "n_estimators": [150, 300],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 3],
    },
    "xgboost": {
        "max_depth": [3, 5],
        "learning_rate": [0.03, 0.1],
        "min_child_weight": [1, 10],
        "scale_pos_weight": [5, 20, 34],
    },
}
