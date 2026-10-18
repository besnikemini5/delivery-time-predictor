import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from . import training
from .data import FEATURES

log = logging.getLogger(__name__)


class DeliveryModel:
    def __init__(self, bundle, metrics):
        self.pipeline = bundle["pipeline"]
        self.error_band = bundle["error_band"]
        self.metrics = metrics

    @classmethod
    def load(cls, model_dir=training.MODEL_DIR):
        model_dir = Path(model_dir)
        bundle = joblib.load(model_dir / training.MODEL_FILE)
        metrics = json.loads((model_dir / training.METRICS_FILE).read_text())
        return cls(bundle, metrics)

    @classmethod
    def load_or_train(cls, model_dir=training.MODEL_DIR):
        model_dir = Path(model_dir)
        if not (model_dir / training.MODEL_FILE).exists():
            log.warning("No trained model found in %s, training one now (takes a few seconds)...", model_dir)
            training.train(model_dir=model_dir)
        return cls.load(model_dir)

    def predict(self, features):
        """Predicted minutes for one delivery, plus a likely range.

        The range covers the middle 80% of the model's own errors on deliveries it
        did not see during training (see training.py).
        """
        X = pd.DataFrame([features], columns=FEATURES)
        minutes = float(self.pipeline.predict(X)[0])
        low, high = training.apply_band(minutes, self.error_band)
        return {"minutes": round(minutes, 1), "low": round(max(float(low), 1.0), 1), "high": round(float(high), 1)}
