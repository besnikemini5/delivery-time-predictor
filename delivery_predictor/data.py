"""Synthetic delivery data.

There is no public dataset with these exact features, so the training data is
generated from a simple model of how a delivery works:

    minutes = handling time (depends on package size)
            + driving time (distance / average speed for the road type)
              x traffic factor x weather factor x time-of-day factor

Random noise is added on top so the data isn't perfectly predictable, the same
way two real deliveries with identical conditions never take exactly as long.
"""
import numpy as np
import pandas as pd

MIN_DISTANCE_KM = 0.5
MAX_DISTANCE_KM = 200

PACKAGE_SIZES = ("small", "medium", "large")
WEATHER = ("clear", "rain", "snow")
TRAFFIC = ("light", "moderate", "heavy")
TIMES_OF_DAY = ("morning", "afternoon", "evening", "night")
ROAD_TYPES = ("highway", "urban", "rural")

CATEGORIES = {
    "package_size": PACKAGE_SIZES,
    "weather": WEATHER,
    "traffic": TRAFFIC,
    "time_of_day": TIMES_OF_DAY,
    "road_type": ROAD_TYPES,
}
FEATURES = ["distance_km", *CATEGORIES]
TARGET = "delivery_minutes"

AVERAGE_SPEED_KMH = {"highway": 80, "urban": 30, "rural": 50}
TRAFFIC_FACTOR = {"light": 1.0, "moderate": 1.3, "heavy": 1.8}
WEATHER_FACTOR = {"clear": 1.0, "rain": 1.2, "snow": 1.6}
TIME_OF_DAY_FACTOR = {"morning": 1.15, "afternoon": 1.0, "evening": 1.2, "night": 0.9}
HANDLING_MINUTES = {"small": 5, "medium": 8, "large": 15}


def expected_minutes(distance_km, road_type, traffic, weather, time_of_day, package_size):
    """Delivery time without noise. Works on single values or pandas Series."""
    def lookup(table, value):
        return value.map(table) if isinstance(value, pd.Series) else table[value]

    driving = (distance_km / lookup(AVERAGE_SPEED_KMH, road_type) * 60
               * lookup(TRAFFIC_FACTOR, traffic)
               * lookup(WEATHER_FACTOR, weather)
               * lookup(TIME_OF_DAY_FACTOR, time_of_day))
    return lookup(HANDLING_MINUTES, package_size) + driving


def generate(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "distance_km": rng.uniform(MIN_DISTANCE_KM, MAX_DISTANCE_KM, n_samples).round(1),
        **{name: rng.choice(values, n_samples) for name, values in CATEGORIES.items()},
    })
    base = expected_minutes(df["distance_km"], df["road_type"], df["traffic"],
                            df["weather"], df["time_of_day"], df["package_size"])
    # About 8% random variation plus a couple of minutes of jitter
    noisy = base * rng.normal(1.0, 0.08, n_samples) + rng.normal(0, 2, n_samples)
    df[TARGET] = np.clip(noisy, 1, None).round(1)
    return df
