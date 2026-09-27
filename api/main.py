from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api.schemas import (
    CustomerFeatures,
    PredictionResponseAPI,
    ExplainabilityResponseAPI,
    DecisionRequest,
    DecisionResponseAPI,
    EconomicDecisionRequest,
    EconomicDecisionResponseAPI,
    DecisionInferenceRequest,
    DecisionInferenceResponseAPI,
)
from project2_core.model_service import ModelService
from project2_core.decision_engine import (
    ECONOMIC_POLICY_VERSION,
    EconomicInputs,
    ThresholdPolicy,
    calculate_net_value,
    economic_decision,
)
from project2_core.policy_config import load_policy_config
from project2_core.logging_config import logger
from project2_core.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    ML_PREDICTIONS_TOTAL,
    ML_DECISIONS_TOTAL,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "model_v7_2.json"
)

POLICY_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "policy_v8_0_4.json"
)


app = FastAPI(
    title="Project 2 — Uplift & Customer Decision API",
    description=(
        "Production-oriented inference API for the frozen "
        "V7.2 Logistic T-learner."
    ),
    version="8.0.4",
)


model_service = ModelService(MODEL_CONFIG_PATH)
policy_config = load_policy_config(POLICY_CONFIG_PATH)


if policy_config.policy_type != "threshold":
    raise ValueError(
        f"Unsupported policy type: {policy_config.policy_type}"
    )


@app.middleware("http")
async def observability_middleware(
    request: Request,
    call_next,
):
    request_id = (
        request.headers.get("X-Request-ID")
        or str(uuid4())
    )

    # Make the request ID available to all downstream handlers.
    request.state.request_id = request_id

    start_time = perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        latency_seconds = perf_counter() - start_time
        latency_ms = latency_seconds * 1000.0

        if request.url.path != "/metrics":
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code="500",
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(latency_seconds)

        logger.exception(
            "request failed",
            extra={
                "event_type": "request_failed",
                "request_id": request_id,
                "http_method": request.method,
                "endpoint": request.url.path,
                "status_code": 500,
                "model_version": model_service.model_version,
                "policy_version": policy_config.policy_version,
                "latency_ms": round(latency_ms, 3),
            },
        )

        raise

    latency_seconds = perf_counter() - start_time
    latency_ms = latency_seconds * 1000.0

    response.headers["X-Request-ID"] = request_id

    if request.url.path != "/metrics":
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=str(response.status_code),
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(latency_seconds)

    logger.info(
        "request completed",
        extra={
            "event_type": "request_completed",
            "request_id": request_id,
            "http_method": request.method,
            "endpoint": request.url.path,
            "status_code": response.status_code,
            "model_version": model_service.model_version,
            "policy_version": policy_config.policy_version,
            "latency_ms": round(latency_ms, 3),
        },
    )

    return response


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": model_service.model_version,
        "policy_version": policy_config.policy_version,
    }


@app.post(
    "/predict",
    response_model=PredictionResponseAPI,
)
def predict(
    customer: CustomerFeatures,
    request: Request,
):
    try:
        result = model_service.predict_one(
            customer.model_dump()
        )

        ML_PREDICTIONS_TOTAL.labels(
            model_version=result.model_version,
        ).inc()

        request_id = request.state.request_id

        logger.info(
            "prediction scored",
            extra={
                "event_type": "prediction_scored",
                "request_id": request_id,
                "endpoint": "/predict",
                "model_version": result.model_version,
                "p_control": result.p_control,
                "p_treatment": result.p_treatment,
                "uplift_score": result.uplift_score,
            },
        )

        return PredictionResponseAPI(
            model_version=result.model_version,
            feature_schema_version=result.feature_schema_version,
            score_version=result.score_version,
            p_control=result.p_control,
            p_treatment=result.p_treatment,
            uplift_score=result.uplift_score,
        )

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post(
    "/explain",
    response_model=ExplainabilityResponseAPI,
)
def explain(
    customer: CustomerFeatures,
    request: Request,
):
    try:
        result = model_service.explain_one(
            customer.model_dump()
        )

        ML_PREDICTIONS_TOTAL.labels(
            model_version=result.model_version,
        ).inc()

        request_id = request.state.request_id

        logger.info(
            "explainability generated",
            extra={
                "event_type": "explainability_generated",
                "request_id": request_id,
                "endpoint": "/explain",
                "model_version": result.model_version,
            },
        )

        return ExplainabilityResponseAPI(
            model_version=result.model_version,
            feature_schema_version=(
                result.feature_schema_version
            ),
            score_version=result.score_version,
            p_control=result.p_control,
            p_treatment=result.p_treatment,
            uplift_score=result.uplift_score,
            control_log_odds=result.control_log_odds,
            treatment_log_odds=result.treatment_log_odds,
            control_intercept=result.control_intercept,
            treatment_intercept=result.treatment_intercept,
            feature_contributions=[
                {
                    "feature": item.feature,
                    "raw_value": item.raw_value,
                    "scaled_value": item.scaled_value,
                    "control_contribution": (
                        item.control_contribution
                    ),
                    "treatment_contribution": (
                        item.treatment_contribution
                    ),
                    "treatment_minus_control_contribution": (
                        item.treatment_minus_control_contribution
                    ),
                }
                for item in result.feature_contributions
            ],
        )

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post(
    "/decide",
    response_model=DecisionResponseAPI,
)
def decide(
    payload: DecisionRequest,
    request: Request,
):
    try:
        policy = ThresholdPolicy(
            policy_version=policy_config.policy_version,
            threshold=payload.threshold,
        )

        decision_value, reason = policy.decide(
            payload.uplift_score
        )

        ML_DECISIONS_TOTAL.labels(
            policy_version=policy_config.policy_version,
            decision=decision_value,
        ).inc()

        request_id = request.state.request_id

        logger.info(
            "decision made",
            extra={
                "event_type": "decision_made",
                "request_id": request_id,
                "endpoint": "/decide",
                "policy_version": policy_config.policy_version,
                "uplift_score": payload.uplift_score,
                "threshold": payload.threshold,
                "decision": decision_value,
            },
        )

        return DecisionResponseAPI(
            decision=decision_value,
            uplift_score=payload.uplift_score,
            threshold=payload.threshold,
            policy_version=policy_config.policy_version,
            reason=reason,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post(
    "/decision/economic",
    response_model=EconomicDecisionResponseAPI,
)
def economic_decide(
    payload: EconomicDecisionRequest,
    request: Request,
):
    try:
        inputs = EconomicInputs(
            treatment_cost=payload.treatment_cost,
            expected_incremental_benefit_per_unit_uplift=(
                payload.expected_incremental_benefit_per_unit_uplift
            ),
            minimum_net_value=payload.minimum_net_value,
        )

        net_value = calculate_net_value(
            payload.uplift_score,
            inputs,
        )

        decision_value, reason = economic_decision(
            payload.uplift_score,
            inputs,
        )

        ML_DECISIONS_TOTAL.labels(
            policy_version=ECONOMIC_POLICY_VERSION,
            decision=decision_value,
        ).inc()

        request_id = request.state.request_id

        logger.info(
            "economic decision made",
            extra={
                "event_type": "economic_decision_made",
                "request_id": request_id,
                "endpoint": "/decision/economic",
                "policy_version": ECONOMIC_POLICY_VERSION,
                "uplift_score": payload.uplift_score,
                "treatment_cost": payload.treatment_cost,
                "expected_incremental_benefit_per_unit_uplift": (
                    payload.expected_incremental_benefit_per_unit_uplift
                ),
                "minimum_net_value": payload.minimum_net_value,
                "net_value": net_value,
                "decision": decision_value,
            },
        )

        return EconomicDecisionResponseAPI(
            policy_type="economic",
            policy_version=ECONOMIC_POLICY_VERSION,
            uplift_score=payload.uplift_score,
            treatment_cost=payload.treatment_cost,
            expected_incremental_benefit_per_unit_uplift=(
                payload.expected_incremental_benefit_per_unit_uplift
            ),
            minimum_net_value=payload.minimum_net_value,
            net_value=net_value,
            decision=decision_value,
            reason=reason,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post(
    "/decision",
    response_model=DecisionInferenceResponseAPI,
)
def decision(
    payload: DecisionInferenceRequest,
    request: Request,
):
    try:
        prediction = model_service.predict_one(
            payload.model_dump(
                exclude={"threshold"}
            )
        )

        ML_PREDICTIONS_TOTAL.labels(
            model_version=prediction.model_version,
        ).inc()

        policy = ThresholdPolicy(
            policy_version=policy_config.policy_version,
            threshold=payload.threshold,
        )

        decision_value, reason = policy.decide(
            prediction.uplift_score
        )

        ML_DECISIONS_TOTAL.labels(
            policy_version=policy_config.policy_version,
            decision=decision_value,
        ).inc()

        request_id = request.state.request_id

        logger.info(
            "prediction and decision completed",
            extra={
                "event_type": "prediction_and_decision",
                "request_id": request_id,
                "endpoint": "/decision",
                "model_version": prediction.model_version,
                "policy_version": policy_config.policy_version,
                "p_control": prediction.p_control,
                "p_treatment": prediction.p_treatment,
                "uplift_score": prediction.uplift_score,
                "threshold": payload.threshold,
                "decision": decision_value,
            },
        )

        return DecisionInferenceResponseAPI(
            model_version=prediction.model_version,
            feature_schema_version=prediction.feature_schema_version,
            score_version=prediction.score_version,
            uplift_score=prediction.uplift_score,
            threshold=payload.threshold,
            decision=decision_value,
            policy_version=policy_config.policy_version,
            reason=reason,
        )

    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
