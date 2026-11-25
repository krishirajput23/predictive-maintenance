"""Model definitions, GPU detection and small hyperparameter searches."""

import subprocess
import time

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import CV_SPLITS, RANDOM_STATE, SEARCH_ITERATIONS, SEARCH_SPACE


def gpu_available():
    """True when an NVIDIA GPU is visible. XGBoost uses it automatically then."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def build_models(scale_pos_weight, use_gpu):
    """Return the baseline, its single-tree comparison and three ensembles."""
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
            ),
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            device="cuda" if use_gpu else "cpu",
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        ),
    }
    return models


def fit_model(name, estimator, X, y, groups, use_gpu):
    """Fit one model, returning (estimator, selected_params, fit_seconds)."""
    search = None
    if name in SEARCH_SPACE:
        cv = StratifiedGroupKFold(
            n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        search = RandomizedSearchCV(
            estimator,
            SEARCH_SPACE[name],
            n_iter=SEARCH_ITERATIONS,
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            refit=True,
            n_jobs=1,
        )

    def run():
        if search is not None:
            search.fit(X, y, groups=groups)
        else:
            estimator.fit(X, y)

    start = time.perf_counter()
    try:
        run()
    except Exception as exc:
        # Some XGBoost wheels are built without CUDA support. Retry on CPU
        # rather than failing the whole experiment.
        if name == "xgboost" and use_gpu and _looks_like_gpu_error(exc):
            print("  GPU training unavailable; retrying XGBoost on CPU.")
            estimator.set_params(device="cpu", tree_method="hist")
            if search is not None:
                search = RandomizedSearchCV(
                    estimator,
                    SEARCH_SPACE[name],
                    n_iter=SEARCH_ITERATIONS,
                    scoring="average_precision",
                    cv=StratifiedGroupKFold(
                        n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
                    ),
                    random_state=RANDOM_STATE,
                    refit=True,
                    n_jobs=1,
                )
            run()
        else:
            raise
    elapsed = time.perf_counter() - start

    if search is not None:
        return search.best_estimator_, search.best_params_, elapsed
    return estimator, {}, elapsed


def _looks_like_gpu_error(exc):
    message = str(exc).lower()
    return any(token in message for token in ("gpu", "cuda", "device"))
