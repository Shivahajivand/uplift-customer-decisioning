from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from .schemas import FEATURES, PredictionResponse


class ModelService:
    """Frozen V7.2 Logistic T-learner serving core. No training or tuning."""

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))

        self.model_version = config["model_version"]
        self.feature_schema_version = "features_v1"
        self.score_version = config["score_version"]

        base = self.config_path.parent.parent
        self.scaler = joblib.load(base / config["scaler_artifact"])
        self.control_model = joblib.load(base / config["control_model_artifact"])
        self.treatment_model = joblib.load(base / config["treatment_model_artifact"])

    @staticmethod
    def _validate_features(features: Mapping[str, float]) -> np.ndarray:
        expected = set(FEATURES)
        received = set(features)

        missing = expected - received
        extra = received - expected
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected features: {sorted(extra)}")

        values = []
        for name in FEATURES:
            value = features[name]
            if not isinstance(value, (int, float, np.number)):
                raise TypeError(f"{name} must be numeric")
            value = float(value)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            values.append(value)

        return np.asarray(values, dtype=float).reshape(1, -1)

    def predict_one(self, features: Mapping[str, float]) -> PredictionResponse:
        x = self._validate_features(features)
        x_scaled = self.scaler.transform(x)

        p_control = float(self.control_model.predict_proba(x_scaled)[0, 1])
        p_treatment = float(self.treatment_model.predict_proba(x_scaled)[0, 1])
        uplift = p_treatment - p_control

        return PredictionResponse(
            model_version=self.model_version,
            feature_schema_version=self.feature_schema_version,
            score_version=self.score_version,
            p_control=p_control,
            p_treatment=p_treatment,
            uplift_score=uplift,
        )
