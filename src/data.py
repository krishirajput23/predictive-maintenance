"""Dataset loading, leakage-safe feature construction and device-grouped splitting."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import (
    DATE,
    DATE_ORIGIN,
    DEVICE,
    DROPPED_METRICS,
    METRICS,
    TARGET,
    TEST_DEVICE_FRACTION,
)


def load_dataset(path):
    """Read and validate the supplied CSV, returning one row per device-day."""
    frame = pd.read_csv(path)
    required = {DATE, DEVICE, TARGET, *METRICS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame[DATE] = pd.to_datetime(frame[DATE], format="%m/%d/%Y")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if not frame[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must be binary (0/1)")
    if frame[[DEVICE, *METRICS]].isna().any().any():
        raise ValueError("Dataset contains missing values in device or metric columns")

    frame = frame.drop_duplicates(subset=[DEVICE, DATE], keep="last")
    return frame.sort_values([DEVICE, DATE], kind="stable").reset_index(drop=True)


def device_failure_labels(frame):
    """A device is positive if it is ever observed with a failure."""
    return frame.groupby(DEVICE)[TARGET].max()


def split_devices(frame, seed=42):
    """Split by device, stratified on the device label.

    Keeping all readings of a device inside one split prevents the model from
    memorising individual machines and gives an honest estimate on unseen
    devices. Returns positional row indices for train, validation and test.
    """
    labels = device_failure_labels(frame)
    devices = labels.index.to_numpy()
    y = labels.to_numpy()

    outer = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_DEVICE_FRACTION, random_state=seed
    )
    train_idx, remaining_idx = next(outer.split(devices, y))

    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = next(
        inner.split(devices[remaining_idx], y[remaining_idx])
    )

    sets = {
        "train": set(devices[train_idx]),
        "val": set(devices[remaining_idx][val_idx]),
        "test": set(devices[remaining_idx][test_idx]),
    }
    if not set.union(*sets.values()) == set(devices):
        raise AssertionError("Device split is not a partition")
    if set.intersection(*sets.values()):
        raise AssertionError("Device splits overlap")

    return {
        name: np.flatnonzero(frame[DEVICE].isin(members).to_numpy())
        for name, members in sets.items()
    }


def build_features(frame):
    """Create numeric features using only information available up to each row.

    Expanding baselines are shifted by one observation so a reading is compared
    against that device's *past* behaviour, never its future or its label.
    """
    features = frame[METRICS].astype(float).copy()
    features = features.drop(columns=DROPPED_METRICS)
    base_metrics = list(features.columns)

    features["day_index"] = (frame[DATE] - DATE_ORIGIN).dt.days.astype(float)
    features["month"] = frame[DATE].dt.month.astype(float)
    features["day_of_week"] = frame[DATE].dt.day_of_week.astype(float)
    features["device_age_days"] = (
        frame[DATE] - frame.groupby(DEVICE)[DATE].transform("min")
    ).dt.days.astype(float)

    for metric in base_metrics:
        history = frame.groupby(DEVICE, sort=False)[metric]
        prior_mean = history.transform(
            lambda values: values.expanding(min_periods=3).mean().shift(1)
        )
        prior_std = history.transform(
            lambda values: values.expanding(min_periods=3).std().shift(1)
        )
        deviation = frame[metric] - prior_mean
        features[f"{metric}_z"] = (deviation / prior_std.replace(0, np.nan)).fillna(0.0)

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features, frame[TARGET].copy(), frame[DEVICE].copy()


def prepare(frame, seed=42):
    """Convenience wrapper used by the training and test suites."""
    features, target, devices = build_features(frame)
    splits = split_devices(frame, seed=seed)
    return features, target, devices, splits
