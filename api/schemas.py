from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """
    API input schema for the 12 baseline Criteo features.

    Pydantic is used at the API boundary.
    The core model service remains framework-independent.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    f0: float = Field(description="Baseline feature f0")
    f1: float = Field(description="Baseline feature f1")
    f2: float = Field(description="Baseline feature f2")
    f3: float = Field(description="Baseline feature f3")
    f4: float = Field(description="Baseline feature f4")
    f5: float = Field(description="Baseline feature f5")
    f6: float = Field(description="Baseline feature f6")
    f7: float = Field(description="Baseline feature f7")
    f8: float = Field(description="Baseline feature f8")
    f9: float = Field(description="Baseline feature f9")
    f10: float = Field(description="Baseline feature f10")
    f11: float = Field(description="Baseline feature f11")


class PredictionResponseAPI(BaseModel):
    """
    Public API response contract for uplift inference.
    """

    model_version: str
    feature_schema_version: str
    score_version: str

    p_control: float = Field(ge=0.0, le=1.0)
    p_treatment: float = Field(ge=0.0, le=1.0)
    uplift_score: float = Field(ge=-1.0, le=1.0)


class ExplainabilityFeatureContributionAPI(BaseModel):
    """
    Per-feature explanation for the frozen Logistic T-learner.

    Contributions are expressed on the model log-odds scale.
    """

    feature: str

    raw_value: float
    scaled_value: float

    control_contribution: float
    treatment_contribution: float
    treatment_minus_control_contribution: float


class ExplainabilityResponseAPI(BaseModel):
    """
    Public API response for model explainability.

    Feature contributions are faithful to the underlying
    LogisticRegression models and are expressed on the
    log-odds scale.
    """

    model_version: str
    feature_schema_version: str
    score_version: str

    p_control: float = Field(ge=0.0, le=1.0)
    p_treatment: float = Field(ge=0.0, le=1.0)
    uplift_score: float = Field(ge=-1.0, le=1.0)

    control_log_odds: float
    treatment_log_odds: float

    control_intercept: float
    treatment_intercept: float

    feature_contributions: list[
        ExplainabilityFeatureContributionAPI
    ]


class DecisionRequest(BaseModel):
    uplift_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Raw uplift score produced by the frozen model.",
    )

    threshold: float = Field(
        ge=-1.0,
        le=1.0,
        description="Decision threshold supplied by the policy configuration.",
    )


class DecisionResponseAPI(BaseModel):
    decision: Literal["TARGET", "DO_NOT_TARGET"]

    uplift_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    threshold: float = Field(
        ge=-1.0,
        le=1.0,
    )

    policy_version: str
    reason: str

class EconomicDecisionRequest(BaseModel):
    """
    Runtime business inputs for cost-sensitive decisioning.
    """

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    uplift_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Raw uplift score produced by the frozen model.",
    )

    treatment_cost: float = Field(
        ge=0.0,
        description="Business cost of applying the treatment.",
    )

    expected_incremental_benefit_per_unit_uplift: float = Field(
        description=(
            "Expected business value associated with one unit "
            "of incremental uplift."
        ),
    )

    minimum_net_value: float = Field(
        description=(
            "Minimum acceptable expected net value for targeting."
        ),
    )


class EconomicDecisionResponseAPI(BaseModel):
    """
    Public API response for cost-sensitive economic decisioning.
    """

    policy_type: Literal["economic"]
    policy_version: str

    uplift_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    treatment_cost: float
    expected_incremental_benefit_per_unit_uplift: float
    minimum_net_value: float

    net_value: float

    decision: Literal["TARGET", "DO_NOT_TARGET"]
    reason: str


class DecisionInferenceRequest(CustomerFeatures):
    """
    End-to-end decision request.

    The client supplies customer features and a policy threshold.
    The uplift score is calculated internally by the model service.
    """

    threshold: float = Field(
        ge=-1.0,
        le=1.0,
        description="Threshold supplied by the configured decision policy.",
    )


class DecisionInferenceResponseAPI(BaseModel):
    """
    Public API response for end-to-end scoring + decisioning.
    """

    model_version: str
    feature_schema_version: str
    score_version: str

    uplift_score: float = Field(
        ge=-1.0,
        le=1.0,
    )

    threshold: float = Field(
        ge=-1.0,
        le=1.0,
    )

    decision: Literal["TARGET", "DO_NOT_TARGET"]

    policy_version: str
    reason: str
