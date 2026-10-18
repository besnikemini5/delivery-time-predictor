"""Trains the delivery time model and saves it together with its test metrics.

Run with:  python -m delivery_predictor.training
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import CATEGORIES, FEATURES, TARGET, generate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
MODEL_FILE = "model.joblib"
METRICS_FILE = "metrics.json"


def build_pipeline(regressor):
    preprocess = ColumnTransformer(
        [("categories", OneHotEncoder(handle_unknown="ignore"), list(CATEGORIES))],
        remainder="passthrough",  # distance_km goes through unchanged
    )
    return Pipeline([("preprocess", preprocess), ("model", regressor)])


def log_target(regressor):
    """Delivery time grows by multiplying factors (traffic x weather x ...), so the
    models learn log(minutes) instead, which turns those products into sums."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def candidates(seed, n_trees):
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": log_target(LinearRegression()),
        "Random forest": RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=3,
                                               random_state=seed, n_jobs=-1),
        "Gradient boosting": log_target(HistGradientBoostingRegressor(random_state=seed)),
    }



def evaluate(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mape": round(float(np.mean(np.abs(y_pred - y_true) / y_true) * 100), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def apply_band(predicted, error_band):
    """Turns point predictions into (low, high) using the log-error percentiles."""
    log_pred = np.log1p(predicted)
    return np.expm1(log_pred + error_band[0]), np.expm1(log_pred + error_band[1])


def feature_importance(pipeline, X_test, y_test, seed):
    """Permutation importance: how much the test error grows when one input is shuffled."""
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=5, random_state=seed,
                                    scoring="neg_mean_absolute_error")
    scores = dict(zip(X_test.columns, np.clip(result.importances_mean, 0, None)))
    total = sum(scores.values()) or 1.0
    return {name: round(float(v / total), 4) for name, v in sorted(scores.items(), key=lambda kv: -kv[1])}


def train(n_samples=5000, seed=42, n_trees=100, model_dir=MODEL_DIR, images_dir=None):
    data = generate(n_samples, seed)
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES], data[TARGET], test_size=0.2, random_state=seed)

    results, fitted = {}, {}
    for name, regressor in candidates(seed, n_trees).items():
        pipeline = build_pipeline(regressor).fit(X_train, y_train)
        results[name] = evaluate(y_test, pipeline.predict(X_test))
        fitted[name] = pipeline

    best_name = min(results, key=lambda name: results[name]["mae"])
    best = fitted[best_name]

    # Likely range: how far off the model was on training deliveries it did not see
    # (5-fold cross-validation), measured as a ratio so it scales with longer trips
    out_of_fold = cross_val_predict(build_pipeline(candidates(seed, n_trees)[best_name]), X_train, y_train, cv=5)
    log_errors = np.log1p(y_train) - np.log1p(out_of_fold)
    error_band = [float(v) for v in np.percentile(log_errors, [10, 90])]
    low, high = apply_band(best.predict(X_test), error_band)
    coverage = float(np.mean((y_test >= low) & (y_test <= high)) * 100)

    metrics = {
        "model": best_name,
        "test_metrics": results[best_name],
        "comparison": results,
        "range": {"error_percentiles": [10, 90], "test_coverage_percent": round(coverage, 1)},
        "feature_importance": feature_importance(best, X_test, y_test, seed),
        "samples": {"train": len(X_train), "test": len(X_test)},
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best, "error_band": error_band}, model_dir / MODEL_FILE, compress=3)
    (model_dir / METRICS_FILE).write_text(json.dumps(metrics, indent=2))

    if images_dir is not None:
        save_plots(best, X_test, y_test, metrics, Path(images_dir))
    return metrics


def save_plots(pipeline, X_test, y_test, metrics, images_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images_dir.mkdir(parents=True, exist_ok=True)
    predicted = pipeline.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, predicted, s=8, alpha=0.4, color="#4361ee")
    limit = [0, max(y_test.max(), predicted.max()) * 1.05]
    ax.plot(limit, limit, "--", color="#e63946", linewidth=1.5, label="Perfect prediction")
    ax.set(xlim=limit, ylim=limit, xlabel="Actual delivery time (min)", ylabel="Predicted delivery time (min)",
           title=f"Test set: off by {metrics['test_metrics']['mape']}% on average, R² {metrics['test_metrics']['r2']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(images_dir / "predicted-vs-actual.png", dpi=120)
    plt.close(fig)

    importance = metrics["feature_importance"]
    if importance:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        labels = [name.replace("_", " ") for name in importance][::-1]
        ax.barh(labels, list(importance.values())[::-1], color="#4361ee")
        ax.set(xlabel="Share of importance (permutation)", title="What drives the prediction")
        fig.tight_layout()
        fig.savefig(images_dir / "feature-importance.png", dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train the delivery time model.")
    parser.add_argument("--samples", type=int, default=5000, help="number of synthetic deliveries to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots", action="store_true", help=f"also save evaluation charts to {IMAGES_DIR}")
    args = parser.parse_args()

    metrics = train(args.samples, args.seed, images_dir=IMAGES_DIR if args.plots else None)
    print(f"Saved {metrics['model']} to {MODEL_DIR / MODEL_FILE}\n")
    print(f"{'Model':<20}{'MAE':>8}{'MAPE %':>8}{'RMSE':>8}{'R²':>8}")
    for name, m in metrics["comparison"].items():
        print(f"{name:<20}{m['mae']:>8}{m['mape']:>8}{m['rmse']:>8}{m['r2']:>8}")
    print(f"\nLikely range contained the real time for "
          f"{metrics['range']['test_coverage_percent']}% of test deliveries")


if __name__ == "__main__":
    main()
