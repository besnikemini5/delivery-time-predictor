import json

import pytest

from delivery_predictor.data import expected_minutes
from delivery_predictor.training import METRICS_FILE, MODEL_FILE

BASE = dict(distance_km=40.0, package_size="medium", weather="clear", traffic="light",
            time_of_day="afternoon", road_type="urban")


def test_training_saves_model_and_metrics(trained_dir):
    assert (trained_dir / MODEL_FILE).exists()
    metrics = json.loads((trained_dir / METRICS_FILE).read_text())
    assert set(metrics["comparison"]) == {"Mean baseline", "Linear regression", "Random forest", "Gradient boosting"}
    assert metrics["samples"] == {"train": 1600, "test": 400}


def test_chosen_model_beats_the_baselines(model):
    comparison = model.metrics["comparison"]
    best = model.metrics["test_metrics"]
    assert best["mae"] < comparison["Mean baseline"]["mae"] / 4
    assert best["mae"] < comparison["Linear regression"]["mae"]
    assert best["r2"] > 0.95
    assert best["mape"] < 15


def test_likely_range_is_honest(model):
    # The range is meant to cover about 80% of deliveries; check it on held-out data
    assert 70 <= model.metrics["range"]["test_coverage_percent"] <= 90


def test_feature_importance_ranks_distance_first(model):
    importance = model.metrics["feature_importance"]
    assert next(iter(importance)) == "distance_km"
    assert sum(importance.values()) == pytest.approx(1, abs=0.01)


def test_prediction_is_close_to_the_true_formula(model):
    predicted = model.predict(BASE)["minutes"]
    truth = expected_minutes(**BASE)
    assert abs(predicted - truth) / truth < 0.2


def test_range_contains_the_prediction(model):
    p = model.predict(BASE)
    assert p["low"] <= p["minutes"] <= p["high"]


@pytest.mark.parametrize("change", [
    {"distance_km": 120.0},
    {"traffic": "heavy"},
    {"weather": "snow"},
])
def test_worse_conditions_mean_longer_deliveries(model, change):
    assert model.predict({**BASE, **change})["minutes"] > model.predict(BASE)["minutes"]


def test_highway_is_faster_than_city_streets(model):
    assert model.predict({**BASE, "road_type": "highway"})["minutes"] < model.predict(BASE)["minutes"]


def test_load_or_train_creates_a_model_when_missing(tmp_path, monkeypatch):
    from delivery_predictor import training
    from delivery_predictor.model import DeliveryModel

    real_train = training.train
    monkeypatch.setattr(training, "train", lambda model_dir: real_train(n_samples=600, n_trees=10, model_dir=model_dir))
    model = DeliveryModel.load_or_train(tmp_path)
    assert (tmp_path / MODEL_FILE).exists()
    assert model.predict(BASE)["minutes"] > 0
