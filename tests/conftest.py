import pytest

from delivery_predictor import create_app
from delivery_predictor.model import DeliveryModel
from delivery_predictor.training import train


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    """Trains one small model for the whole test run (takes a few seconds)."""
    model_dir = tmp_path_factory.mktemp("model")
    train(n_samples=2000, seed=7, n_trees=50, model_dir=model_dir)
    return model_dir


@pytest.fixture(scope="session")
def model(trained_dir):
    return DeliveryModel.load(trained_dir)


@pytest.fixture
def client(model):
    return create_app(model=model).test_client()


VALID = {
    "distance_km": "40",
    "package_size": "medium",
    "weather": "clear",
    "traffic": "light",
    "time_of_day": "afternoon",
    "road_type": "urban",
}
