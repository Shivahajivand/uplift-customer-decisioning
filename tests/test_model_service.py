import hashlib
import json
from pathlib import Path

import pytest

from project2_core.model_service import ModelService


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "model_v7_2.json"
)


@pytest.fixture(scope="module")
def model_service():
    return ModelService(MODEL_CONFIG)


def test_model_service_loads(model_service):
    assert model_service.model_version == "v7.2"
    assert model_service.feature_schema_version == "features_v1"
    assert model_service.score_version == "raw_uplift_v1"


def test_model_artifact_hashes_match_manifest(model_service):
    config = json.loads(
        MODEL_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    base = MODEL_CONFIG.parent.parent

    artifact_paths = {
        "scaler": (
            base / config["scaler_artifact"]
        ),
        "control_model": (
            base
            / config["control_model_artifact"]
        ),
        "treatment_model": (
            base
            / config["treatment_model_artifact"]
        ),
    }

    expected_hashes = config["artifact_sha256"]

    for artifact_name, path in artifact_paths.items():
        digest = hashlib.sha256()

        with path.open("rb") as file:
            for chunk in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        actual_hash = digest.hexdigest().upper()

        assert actual_hash == (
            expected_hashes[artifact_name].upper()
        )


def test_model_service_rejects_artifact_hash_mismatch(
    model_service,
):
    scaler_path = (
        PROJECT_ROOT
        / "models"
        / "v7.2"
        / "scaler_v7_2.joblib"
    )

    with pytest.raises(
        ValueError,
        match="Model artifact integrity check failed",
    ):
        ModelService._verify_artifact(
            "scaler",
            scaler_path,
            "0" * 64,
        )


def test_prediction_contract(model_service):
    features = {
        f"f{i}": 0.0
        for i in range(12)
    }

    result = model_service.predict_one(features)

    assert 0.0 <= result.p_control <= 1.0
    assert 0.0 <= result.p_treatment <= 1.0
    assert -1.0 <= result.uplift_score <= 1.0

    assert result.uplift_score == pytest.approx(
        result.p_treatment - result.p_control
    )


def test_prediction_is_deterministic(model_service):
    features = {
        f"f{i}": 0.0
        for i in range(12)
    }

    result_1 = model_service.predict_one(features)
    result_2 = model_service.predict_one(features)

    assert result_1.p_control == pytest.approx(
        result_2.p_control
    )

    assert result_1.p_treatment == pytest.approx(
        result_2.p_treatment
    )

    assert result_1.uplift_score == pytest.approx(
        result_2.uplift_score
    )
