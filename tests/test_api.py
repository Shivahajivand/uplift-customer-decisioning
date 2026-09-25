import logging
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from api.main import app
from project2_core.logging_config import logger


client = TestClient(app)


VALID_PAYLOAD = {
    "f0": 0.0,
    "f1": 0.0,
    "f2": 0.0,
    "f3": 0.0,
    "f4": 0.0,
    "f5": 0.0,
    "f6": 0.0,
    "f7": 0.0,
    "f8": 0.0,
    "f9": 0.0,
    "f10": 0.0,
    "f11": 0.0,
}


@pytest.fixture
def captured_project2_logs():
    records = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = CaptureHandler()
    logger.addHandler(handler)

    try:
        yield records
    finally:
        logger.removeHandler(handler)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_version"] == "v7.2"


def test_predict_valid_request():
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "v7.2"
    assert body["feature_schema_version"] == "features_v1"
    assert body["score_version"] == "raw_uplift_v1"

    assert 0.0 <= body["p_control"] <= 1.0
    assert 0.0 <= body["p_treatment"] <= 1.0
    assert -1.0 <= body["uplift_score"] <= 1.0


def test_predict_rejects_extra_feature():
    payload = {
        **VALID_PAYLOAD,
        "f12": 0.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_missing_feature():
    payload = VALID_PAYLOAD.copy()
    payload.pop("f11")

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_wrong_type():
    payload = {
        **VALID_PAYLOAD,
        "f3": "hello",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_nan():
    payload = {
        **VALID_PAYLOAD,
        "f3": "NaN",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_decide_target():
    response = client.post(
        "/decide",
        json={
            "uplift_score": 0.05,
            "threshold": 0.01,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["decision"] == "TARGET"
    assert body["uplift_score"] == 0.05
    assert body["threshold"] == 0.01
    assert body["policy_version"] == "v8.0.4-threshold-v1"


def test_decide_do_not_target():
    response = client.post(
        "/decide",
        json={
            "uplift_score": 0.005,
            "threshold": 0.01,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["decision"] == "DO_NOT_TARGET"
    assert body["uplift_score"] == 0.005
    assert body["threshold"] == 0.01
    assert body["policy_version"] == "v8.0.4-threshold-v1"


def test_end_to_end_decision_do_not_target():
    payload = {
        **VALID_PAYLOAD,
        "threshold": 0.01,
    }

    response = client.post("/decision", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "v7.2"
    assert body["decision"] == "DO_NOT_TARGET"
    assert body["uplift_score"] == 0.0
    assert body["threshold"] == 0.01
    assert body["policy_version"] == "v8.0.4-threshold-v1"


def test_end_to_end_decision_target():
    payload = {
        **VALID_PAYLOAD,
        "threshold": 0.0,
    }

    response = client.post("/decision", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "v7.2"
    assert body["decision"] == "TARGET"
    assert body["uplift_score"] == 0.0
    assert body["threshold"] == 0.0


def test_decision_rejects_threshold_above_max():
    payload = {
        **VALID_PAYLOAD,
        "threshold": 1.1,
    }

    response = client.post("/decision", json=payload)

    assert response.status_code == 422


def test_decision_rejects_threshold_below_min():
    payload = {
        **VALID_PAYLOAD,
        "threshold": -1.1,
    }

    response = client.post("/decision", json=payload)

    assert response.status_code == 422


def test_client_cannot_supply_uplift_score():
    payload = {
        **VALID_PAYLOAD,
        "threshold": 0.01,
        "uplift_score": 0.99,
    }

    response = client.post("/decision", json=payload)

    assert response.status_code == 422


def test_observability_preserves_request_id():
    response = client.get(
        "/health",
        headers={"X-Request-ID": "test-request-123"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-123"


def test_observability_generates_request_id():
    response = client.get("/health")

    assert response.status_code == 200

    request_id = response.headers.get("X-Request-ID")

    assert request_id is not None

    UUID(request_id)


def test_predict_structured_log_contains_ml_event_without_features(
    captured_project2_logs,
):
    response = client.post(
        "/predict",
        json=VALID_PAYLOAD,
        headers={"X-Request-ID": "test-predict-log-001"},
    )

    assert response.status_code == 200

    events = [
        record
        for record in captured_project2_logs
        if getattr(record, "event_type", None)
        == "prediction_scored"
    ]

    assert len(events) == 1

    record = events[0]

    assert record.request_id == "test-predict-log-001"
    assert record.endpoint == "/predict"
    assert record.model_version == "v7.2"

    assert hasattr(record, "p_control")
    assert hasattr(record, "p_treatment")
    assert hasattr(record, "uplift_score")

    for feature in (f"f{i}" for i in range(12)):
        assert not hasattr(record, feature)


def test_decide_structured_log_contains_decision_event_without_features(
    captured_project2_logs,
):
    response = client.post(
        "/decide",
        json={
            "uplift_score": 0.026,
            "threshold": 0.01,
        },
        headers={"X-Request-ID": "test-decide-log-001"},
    )

    assert response.status_code == 200

    events = [
        record
        for record in captured_project2_logs
        if getattr(record, "event_type", None)
        == "decision_made"
    ]

    assert len(events) == 1

    record = events[0]

    assert record.request_id == "test-decide-log-001"
    assert record.endpoint == "/decide"
    assert record.policy_version == "v8.0.4-threshold-v1"

    assert record.uplift_score == 0.026
    assert record.threshold == 0.01
    assert record.decision == "TARGET"

    for feature in (f"f{i}" for i in range(12)):
        assert not hasattr(record, feature)


def test_end_to_end_structured_log_contains_prediction_and_decision_event_without_features(
    captured_project2_logs,
):
    payload = {
        **VALID_PAYLOAD,
        "threshold": 0.01,
    }

    response = client.post(
        "/decision",
        json=payload,
        headers={"X-Request-ID": "test-e2e-log-001"},
    )

    assert response.status_code == 200

    events = [
        record
        for record in captured_project2_logs
        if getattr(record, "event_type", None)
        == "prediction_and_decision"
    ]

    assert len(events) == 1

    record = events[0]

    assert record.request_id == "test-e2e-log-001"
    assert record.endpoint == "/decision"
    assert record.model_version == "v7.2"
    assert record.policy_version == "v8.0.4-threshold-v1"

    assert hasattr(record, "p_control")
    assert hasattr(record, "p_treatment")
    assert hasattr(record, "uplift_score")

    assert record.threshold == 0.01
    assert record.decision == "DO_NOT_TARGET"

    for feature in (f"f{i}" for i in range(12)):
        assert not hasattr(record, feature)
