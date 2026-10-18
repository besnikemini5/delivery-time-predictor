import pytest

from tests.conftest import VALID


def test_home_page(client):
    page = client.get("/").get_data(as_text=True)
    assert "Delivery Time Predictor" in page
    assert 'name="distance_km"' in page


def test_form_prediction(client):
    response = client.post("/predict", data=VALID)
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "Estimated delivery time" in page
    assert "Most likely between" in page
    assert "City streets" in page  # the inputs are shown back in plain words


@pytest.mark.parametrize("field, value, message", [
    ("distance_km", "abc", "Enter the distance as a number"),
    ("distance_km", "0", "between 0.5 and 200"),
    ("distance_km", "5000", "between 0.5 and 200"),
    ("distance_km", "nan", "between 0.5 and 200"),
    ("weather", "tornado", "Please choose an option"),
    ("traffic", "", "Please choose an option"),
])
def test_invalid_form_input_shows_an_error(client, field, value, message):
    response = client.post("/predict", data={**VALID, field: value})
    assert response.status_code == 400
    page = response.get_data(as_text=True)
    assert message in page
    assert "Please fix the highlighted fields" in page


def test_form_keeps_valid_values_after_an_error(client):
    page = client.post("/predict", data={**VALID, "distance_km": "abc"}).get_data(as_text=True)
    assert '<option value="urban" selected>' in page


def test_api_prediction(client):
    response = client.post("/api/predict", json={**VALID, "distance_km": 40})
    assert response.status_code == 200
    body = response.get_json()
    assert body["predicted_minutes"] > 0
    assert body["likely_range"]["low"] <= body["predicted_minutes"] <= body["likely_range"]["high"]
    assert body["inputs"]["road_type"] == "urban"
    assert body["model"]["name"] == "Gradient boosting"


def test_api_accepts_mixed_case_options(client):
    assert client.post("/api/predict", json={**VALID, "weather": " Rain "}).status_code == 200


@pytest.mark.parametrize("payload", [
    {**VALID, "distance_km": -3},
    {**VALID, "distance_km": True},
    {**VALID, "distance_km": None},
    {**VALID, "road_type": ["urban"]},
    {k: v for k, v in VALID.items() if k != "traffic"},
])
def test_api_rejects_invalid_input(client, payload):
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "fields" in response.get_json()


def test_api_rejects_non_object_body(client):
    assert client.post("/api/predict", json=[1, 2]).status_code == 400
    assert client.post("/api/predict", data="hello").status_code == 400


def test_api_model_metrics(client):
    body = client.get("/api/model").get_json()
    assert body["model"] == "Gradient boosting"
    assert "test_metrics" in body


def test_model_page(client):
    page = client.get("/model").get_data(as_text=True)
    assert "Model comparison" in page
    assert "Gradient boosting" in page
    assert "synthetic" in page


def test_switch_to_spanish(client):
    response = client.get("/language/es", headers={"Referer": "http://localhost/model"})
    assert response.status_code == 302
    assert response.headers["Location"] == "/model"
    page = client.get("/").get_data(as_text=True)
    assert "Predictor de tiempo de entrega" in page
    assert 'lang="es"' in page


def test_spanish_result_page(client):
    client.get("/language/es")
    page = client.post("/predict", data=VALID).get_data(as_text=True)
    assert "Tiempo de entrega estimado" in page


def test_language_switch_never_redirects_to_another_site(client):
    response = client.get("/language/es", headers={"Referer": "https://evil.example.com/phish"})
    assert response.headers["Location"] == "/"


def test_unknown_language_returns_404(client):
    assert client.get("/language/xx").status_code == 404


def test_unknown_page_returns_404(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert "does not exist" in response.get_data(as_text=True)


def test_unknown_api_route_returns_json(client):
    response = client.get("/api/nope")
    assert response.status_code == 404
    assert response.is_json
