"""Paths, constants, feature definitions, and experiment configs."""

from datetime import timedelta
from pathlib import Path

import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[0]

DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "athletes.csv"
CLEAN_PARQUET = DATA_DIR / "athletes_clean.parquet"

FEATURE_REPO_DIR = PROJECT_ROOT / "feature_repo"
FEATURE_DATA_DIR = FEATURE_REPO_DIR / "data"
FEATURES_V1_PARQUET = FEATURE_DATA_DIR / "athlete_features_v1.parquet"
FEATURES_V2_PARQUET = FEATURE_DATA_DIR / "athlete_features_v2.parquet"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"
MLFLOW_EXPERIMENT_NAME = "athlete_total_lift"

# Reproducibility
SEED = 42
TEST_SIZE = 0.2

# Target definition
TARGET = "total_lift"
LIFT_COMPONENTS = ["deadlift", "candj", "snatch", "backsq"]

# Entity / timestamp columns
ENTITY_KEY = "athlete_id"
EVENT_TIMESTAMP = "event_timestamp"

# Feature snapshot timestamp; entity rows are dated one day later so the Feast
# point-in-time join returns the features (within their TTL).
FEATURE_TIMESTAMP = pd.Timestamp("2021-01-01", tz="UTC")
ENTITY_TIMESTAMP = FEATURE_TIMESTAMP + timedelta(days=1)

# Feature versions
# v1: baseline
FEATURES_V1 = ["age", "height", "weight", "gender_male"]

# v2: v1 plus engineered features
FEATURES_V2 = [
    *FEATURES_V1,
    "bmi",
    "weight_to_height",
    "age_bucket",
    "is_experienced",
]

FEATURE_VERSIONS = {
    "v1": FEATURES_V1,
    "v2": FEATURES_V2,
}

# Feast feature-view names per version
FEAST_FEATURE_VIEWS = {
    "v1": "athlete_features_v1",
    "v2": "athlete_features_v2",
}

# Hyperparameter configs (XGBoost)
HYPERPARAM_CONFIGS = {
    "shallow": {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "deep": {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
}

# Static XGBoost params
XGB_STATIC_PARAMS = {
    "objective": "reg:squarederror",
    "random_state": SEED,
    "n_jobs": -1,
}