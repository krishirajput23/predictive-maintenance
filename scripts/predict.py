"""Score device readings with the saved model (working prototype).

Usage:
  python scripts/predict.py --input readings.csv [--output scores.csv]
  python scripts/predict.py                       # demo on supplied readings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config
from src.data import build_features


def read_input(path):
    frame = pd.read_csv(path)
    required = {config.DEVICE, config.DATE, *config.METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    frame[config.DATE] = pd.to_datetime(frame[config.DATE], format="%m/%d/%Y")
    return frame


def score(frame, bundle):
    features, _, _ = build_features(frame)
    features = features.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    probability = bundle["model"].predict_proba(features)[:, 1]
    threshold = bundle["threshold"]

    result = pd.DataFrame({
        "device": frame[config.DEVICE],
        "date": frame[config.DATE],
        "failure_probability": probability,
        "decision": ["MAINTENANCE" if p >= threshold else "OK"
                     for p in probability],
    })
    if config.TARGET in frame.columns:
        result["actual_failure"] = frame[config.TARGET].to_numpy()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        help="CSV with device, date (m/d/YYYY) and metric1..metric9")
    parser.add_argument("--output", type=Path, help="optional CSV to save the scores")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if args.input is None:
        full = read_input(config.DATA_PATH)
        failing = full.loc[full[config.TARGET] == 1, config.DEVICE].drop_duplicates().head(2)
        readings = full[full[config.DEVICE].isin(failing)]
        scores = score(readings, bundle)
        print("Demo mode: last readings of two devices that failed in the dataset.")
        print(scores.groupby(config.DEVICE).tail(3).to_string(index=False))
    else:
        readings = read_input(args.input)
        scores = score(readings, bundle)
        print(scores.to_string(index=False))

    print(f"\nDecision threshold: {bundle['threshold']:.4f}")
    if args.output:
        scores.to_csv(args.output, index=False)
        print(f"Scores saved to {args.output}")


if __name__ == "__main__":
    main()
