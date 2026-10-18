import math

from .data import CATEGORIES, MAX_DISTANCE_KM, MIN_DISTANCE_KM


def parse_inputs(source):
    """Validates form or JSON input.

    Returns (features, errors): features is a dict ready for the model, errors maps
    field names to a message key (translated in the template).
    """
    features, errors = {}, {}

    raw = source.get("distance_km")
    try:
        if isinstance(raw, bool):
            raise ValueError
        distance = float(raw)
        if not math.isfinite(distance) or not MIN_DISTANCE_KM <= distance <= MAX_DISTANCE_KM:
            errors["distance_km"] = "error_distance_range"
        else:
            features["distance_km"] = round(distance, 1)
    except (TypeError, ValueError):
        errors["distance_km"] = "error_distance_number"

    for field, allowed in CATEGORIES.items():
        value = source.get(field)
        if isinstance(value, str) and value.strip().lower() in allowed:
            features[field] = value.strip().lower()
        else:
            errors[field] = "error_choose_option"

    return features, errors
