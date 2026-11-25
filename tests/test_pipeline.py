"""Small smoke tests for the data pipeline and the saved prototype."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.predict import score
from src import config
from src.data import build_features, load_dataset, prepare


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = load_dataset(config.DATA_PATH)

    def test_dataset_is_valid(self):
        self.assertGreater(len(self.frame), 100_000)
        self.assertTrue({config.DATE, config.DEVICE, config.TARGET}.issubset(self.frame.columns))
        self.assertTrue(set(self.frame[config.TARGET].unique()).issubset({0, 1}))
        self.assertFalse(self.frame.duplicated([config.DEVICE, config.DATE]).any())

    def test_features_are_finite(self):
        features, _, _ = build_features(self.frame)
        self.assertFalse(features.isna().any().any())
        self.assertTrue(np.isfinite(features.to_numpy()).all())
        self.assertNotIn("metric8", features.columns)  # duplicate of metric7

    def test_split_is_grouped_and_stratified(self):
        _, y, devices, splits = prepare(self.frame, seed=config.RANDOM_STATE)
        sets = [set(devices.iloc[idx]) for idx in splits.values()]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertFalse(sets[i] & sets[j])
        for idx in splits.values():
            self.assertGreater(y.iloc[idx].sum(), 0)

    def test_saved_model_scores_readings(self):
        if not config.MODEL_PATH.exists():
            self.skipTest("Run scripts/train.py first")
        import joblib

        bundle = joblib.load(config.MODEL_PATH)
        failing = self.frame.loc[self.frame[config.TARGET] == 1].head(1)
        result = score(failing, bundle)
        self.assertEqual(len(result), 1)
        self.assertIn(result.iloc[0]["decision"], {"OK", "MAINTENANCE"})
        self.assertGreaterEqual(result.iloc[0]["failure_probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
