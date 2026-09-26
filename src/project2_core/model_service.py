from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from .schemas import FEATURES, PredictionResponse


class ModelService:
    """Frozen V7.2 Logistic T-learner serving core.

    No training or tuning is performed at serving time.
    Model artifacts are verified against their SHA-256
    checksums before loading.
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)

        config = json.loads(
            self.config_path.read_text(
                encoding="utf-8"
            )
        )

        self.model_version = config["model_version"]
        self.feature_schema_version = "features_v1"
        self.score_version = config["score_version"]

        self.artifact_sha256 = config["artifact_sha256"]

        base = self.config_path.parent.parent

        self.scaler_path = (
            base / config["scaler_artifact"]
        )
        self.control_model_path = (
            base / config["control_model_artifact"]
        )
        self.treatment_model_path = (
            base / config["treatment_model_artifact"]
        )

        self._verify_artifact(
            "scaler",
            self.scaler_path,
            self.artifact_sha256["scaler"],
        )

        self._verify_artifact(
            "control_model",
            self.control_model_path,
            self.artifact_sha256["control_model"],
        )

        self._verify_artifact(
            "treatment_model",
            self.treatment_model_path,
            self.artifact_sha256["treatment_model"],
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

        self.control_model = joblib.load(
            self.control_model_path
        )

        self.treatment_model = joblib.load(
            self.treatment_model_path
        )

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest().upper()

    @classmethod
    def _verify_artifact(
        cls,
        artifact_name: str,
        path: Path,
        expected_hash: str,
    ) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required model artifact not found: {path}"
            )

        actual_hash = cls._sha256_file(path)

        if actual_hash != expected_hash.upper():
            raise ValueError(
                "Model artifact integrity check failed "
                f"for '{artifact_name}': "
                f"expected SHA-256 {expected_hash.upper()}, "
                f"got {actual_hash}"
            )

    @staticmethod
    def _validate_features(
        features: Mapping[str, float],
    ) -> np.ndarray:
        expected = set(FEATURES)
        received = set(features)

        missing = expected - received
        extra = received - expected

        if missing:
            raise ValueError(
                f"Missing features: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Unexpected features: {sorted(extra)}"
            )

        values = []

        for name in FEATURES:
            value = features[name]

            if not isinstance(
                value,
                (int, float, np.number),
            ):
                raise TypeError(
                    f"{name} must be numeric"
                )

            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite"
                )

            values.append(value)

        return np.asarray(
            values,
            dtype=float,
        ).reshape(1, -1)

    def predict_one(
        self,
        features: Mapping[str, float],
    ) -> PredictionResponse:
        x = self._validate_features(features)

        x_scaled = self.scaler.transform(x)

        p_control = float(
            self.control_model.predict_proba(
                x_scaled
            )[0, 1]
        )

        p_treatment = float(
            self.treatment_model.predict_proba(
                x_scaled
            )[0, 1]
        )

        uplift = p_treatment - p_control

        return PredictionResponse(
            model_version=self.model_version,
            feature_schema_version=self.feature_schema_version,
            score_version=self.score_version,
            p_control=p_control,
            p_treatment=p_treatment,
            uplift_score=uplift,
        )
