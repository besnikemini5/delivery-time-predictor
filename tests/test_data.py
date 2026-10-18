from delivery_predictor.data import (CATEGORIES, FEATURES, MAX_DISTANCE_KM, MIN_DISTANCE_KM, TARGET,
                                     expected_minutes, generate)


def test_generated_data_has_the_expected_shape():
    df = generate(500, seed=1)
    assert list(df.columns) == FEATURES + [TARGET]
    assert len(df) == 500
    assert df["distance_km"].between(MIN_DISTANCE_KM, MAX_DISTANCE_KM).all()
    assert (df[TARGET] >= 1).all()
    for column, allowed in CATEGORIES.items():
        assert set(df[column]) <= set(allowed)


def test_generation_is_reproducible():
    assert generate(200, seed=3).equals(generate(200, seed=3))
    assert not generate(200, seed=3).equals(generate(200, seed=4))


def test_formula_behaves_like_real_deliveries():
    base = dict(distance_km=50, road_type="urban", traffic="light", weather="clear",
                time_of_day="afternoon", package_size="small")
    t = expected_minutes(**base)
    assert expected_minutes(**{**base, "distance_km": 100}) > t
    assert expected_minutes(**{**base, "traffic": "heavy"}) > t
    assert expected_minutes(**{**base, "weather": "snow"}) > t
    assert expected_minutes(**{**base, "road_type": "highway"}) < t
    assert expected_minutes(**{**base, "time_of_day": "night"}) < t
    assert expected_minutes(**{**base, "package_size": "large"}) > t


def test_formula_example():
    # 30 km on city streets at 30 km/h = 60 min driving, plus 5 min handling
    assert expected_minutes(30, "urban", "light", "clear", "afternoon", "small") == 65
