from dataclasses import dataclass
from typing import Literal, Optional

FEATURES = tuple(f"f{i}" for i in range(12))

@dataclass(frozen=True)
class PredictionResponse:
    model_version: str
    feature_schema_version: str
    score_version: str
    p_control: float
    p_treatment: float
    uplift_score: float

@dataclass(frozen=True)
class DecisionResponse:
    model_version: str
    policy_version: str
    policy_type: str
    decision: Literal["TARGET", "DO_NOT_TARGET"]
    uplift_score: float
    reason: str
    threshold: Optional[float] = None
