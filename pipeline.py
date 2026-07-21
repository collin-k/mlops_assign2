"""End-to-end feature-store ML pipeline

Run:
    python pipeline.py
"""

from __future__ import annotations

import os
import subprocess

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from feast import FeatureStore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import config

def load_and_clean() -> pd.DataFrame:
    """Load CSV and return a cleaned athletes DataFrame."""

    df = pd.read_csv(config.RAW_CSV)

    # Limit to relevant columns (keep athlete_id as the Feast entity key)
    df = df[[config.ENTITY_KEY, 'gender', 'age', 'height', 'weight', 'howlong', 'background', *config.LIFT_COMPONENTS]].copy()
    
    # Remove outliers
    df = df[df['weight'] < 1500]
    df = df[df['gender'] != '--']
    df = df[df['age'] >= 18]
    df = df[(df['height'] < 96) & (df['height'] > 48)]

    df = df[
        ((df['gender'] == 'Male') & (df['deadlift'] <= 1105)) |
        ((df['gender'] == 'Female') & (df['deadlift'] <= 636))
    ]
    df = df[(df['candj'] > 0) & (df['candj'] <= 395)]
    df = df[(df['snatch'] > 0) & (df['snatch'] <= 496)]
    df = df[(df['backsq'] > 0) & (df['backsq'] <= 1069)]

    # Clean survey data
    decline_dict = {'Decline to answer|': np.nan}
    df = df.replace(decline_dict)

    # Remove missing values
    df = df.dropna().copy()

    df[config.TARGET] = df[config.LIFT_COMPONENTS].sum(axis=1)

    return df


def build_features(clean_df: pd.DataFrame) -> None:
    """Build v1 (baseline) and v2 (engineered) feature tables."""
    config.FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_df.copy()

    # Feast requires event timestamp
    df[config.EVENT_TIMESTAMP] = config.FEATURE_TIMESTAMP

    # Baseline features for v1 and v2
    df["gender_male"] = (df["gender"] == "Male").astype("int64")

    # Additional features for v2
    df["bmi"] = (df["weight"] / (df["height"] ** 2) * 703).round(3)
    df["weight_to_height"] = (df["weight"] / df["height"]).round(3)
    df["age_bucket"] = pd.cut(df["age"], [18, 25, 32, 39, 46, 60], labels=False, include_lowest=True).astype("int64")
    df["is_experienced"] = df["howlong"].fillna("").str.lower().str.contains("2-4 years|4+ years", regex=False).astype("int64")

    for version, features in config.FEATURE_VERSIONS.items():
        cols = [config.ENTITY_KEY, config.EVENT_TIMESTAMP, *features]
        out = config.FEATURE_DATA_DIR / f"{config.FEAST_FEATURE_VIEWS[version]}.parquet"
        df[cols].to_parquet(out, index=False)
        print(f"  wrote {version}: {len(features)} features -> {out.name}")


def feast_apply() -> None:
    """Register the entity and both feature views with Feast."""
    subprocess.run(["feast", "apply"], cwd=str(config.FEATURE_REPO_DIR), check=True)


def get_training_data(version: str, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieve a point-in-time training set from Feast."""
    store = FeatureStore(repo_path=str(config.FEATURE_REPO_DIR))
    entity_df = clean_df[[config.ENTITY_KEY, config.TARGET]].copy()
    entity_df[config.EVENT_TIMESTAMP] = config.ENTITY_TIMESTAMP

    refs = [f"{config.FEAST_FEATURE_VIEWS[version]}:{f}" for f in config.FEATURE_VERSIONS[version]]
    training_df = store.get_historical_features(entity_df=entity_df, features=refs).to_df()
    return training_df[config.FEATURE_VERSIONS[version] + [config.TARGET]]


def _save_pred_plot(y_true, y_pred, run_name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.FIGURES_DIR / f"{run_name}_pred_vs_actual.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=8, alpha=0.3, edgecolor="none")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", label="ideal (y = x)")
    ax.set(xlabel="Actual total_lift (lb)", ylabel="Predicted total_lift (lb)", title=run_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def train_one(version: str, cfg_name: str, clean_df: pd.DataFrame) -> dict:
    """Run and log a single (feature version, hyperparameter) experiment."""
    run_name = f"{version}_{cfg_name}"
    features = config.FEATURE_VERSIONS[version]
    hyperparams = config.HYPERPARAM_CONFIGS[cfg_name]
    data = get_training_data(version, clean_df)

    X_train, X_test, y_train, y_test = train_test_split(
        data[features], data[config.TARGET], test_size=config.TEST_SIZE, random_state=config.SEED
    )

    model = XGBRegressor(**config.XGB_STATIC_PARAMS, **hyperparams)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({"feature_version": version, "hyperparam_config": cfg_name,
                         "algorithm": "XGBoostRegressor"})
        mlflow.log_params({"feature_version": version, "hyperparam_config": cfg_name,
                           "n_features": len(features), **hyperparams})

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(_save_pred_plot(y_test, y_pred, run_name)), "figures")
        mlflow.xgboost.log_model(model, artifact_path="model")

    print(f"  [{run_name}] RMSE={metrics['rmse']:.2f}  MAE={metrics['mae']:.2f}  R2={metrics['r2']:.4f}")
    return {"version": version, "config": cfg_name, **metrics}


def _save_comparison_plot(results: pd.DataFrame) -> None:
    labels = results["version"] + "\n" + results["config"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(axes, ["rmse", "mae", "r2"],
                                 ["RMSE", "MAE", "R2"]):
        ax.bar(labels, results[metric], color="#4C72B0")
        ax.set(title=title, ylabel=metric.upper())
    fig.suptitle("4 experiments: 2 feature versions x 2 hyperparameter configs")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "experiment_comparison.png", dpi=120)
    plt.close(fig)


def main() -> pd.DataFrame:

    print("== Ingest + preprocess ==")
    clean_df = load_and_clean()

    print("== Feature engineering ==")
    build_features(clean_df)

    print("== Feast apply ==")
    feast_apply()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    print("== Experiments ==")
    results = pd.DataFrame(
        train_one(v, c, clean_df)
        for v in config.FEATURE_VERSIONS
        for c in config.HYPERPARAM_CONFIGS
    )

    results.to_csv(config.REPORTS_DIR / "experiment_results.csv", index=False)
    _save_comparison_plot(results)

    print("\n== Summary ==")
    print(results.to_string(index=False))
    print("\nBrowse runs:  mlflow ui --backend-store-uri ./mlruns")
    return results


if __name__ == "__main__":
    main()
