from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdPolicy:
    """
    Generic threshold-based decision policy.

    This policy converts a pre-computed uplift score into a business
    action using an externally supplied threshold.

    The threshold is a policy input. It is NOT claimed to be optimal
    based on V7.9 Final Validation.
    """

    policy_version: str
    threshold: float

    def decide(self, uplift_score: float) -> tuple[str, str]:
        if not (-1.0 <= uplift_score <= 1.0):
            raise ValueError("uplift_score must be in [-1, 1].")

        if not (-1.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [-1, 1].")

        if uplift_score >= self.threshold:
            return (
                "TARGET",
                f"uplift_score >= configured threshold ({self.threshold:.8f})",
            )

        return (
            "DO_NOT_TARGET",
            f"uplift_score < configured threshold ({self.threshold:.8f})",
        )


@dataclass(frozen=True)
class CapacityPolicy:
    """
    Threshold primitive for population-level capacity decisions.

    A batch system derives the threshold from the target population,
    then supplies that threshold to the online decision function.
    """

    policy_version: str
    threshold: float

    def decide(self, uplift_score: float) -> tuple[str, str]:
        if not (-1.0 <= uplift_score <= 1.0):
            raise ValueError("uplift_score must be in [-1, 1].")

        if uplift_score >= self.threshold:
            return (
                "TARGET",
                f"uplift_score >= configured threshold ({self.threshold:.8f})",
            )

        return (
            "DO_NOT_TARGET",
            f"uplift_score < configured threshold ({self.threshold:.8f})",
        )


@dataclass(frozen=True)
class EconomicInputs:
    treatment_cost: float
    expected_incremental_benefit_per_unit_uplift: float
    minimum_net_value: float = 0.0


def economic_decision(
    uplift_score: float,
    inputs: EconomicInputs,
) -> tuple[str, str]:
    net_value = (
        uplift_score * inputs.expected_incremental_benefit_per_unit_uplift
        - inputs.treatment_cost
    )

    if net_value >= inputs.minimum_net_value:
        return (
            "TARGET",
            f"net_value={net_value:.8f} >= minimum_net_value",
        )

    return (
        "DO_NOT_TARGET",
        f"net_value={net_value:.8f} < minimum_net_value",
    )


def derive_capacity_threshold(
    uplift_scores,
    capacity_fraction: float,
) -> float:
    import numpy as np

    if not (0 < capacity_fraction <= 1):
        raise ValueError("capacity_fraction must be in (0, 1].")

    scores = np.asarray(list(uplift_scores), dtype=float)

    if scores.size == 0:
        raise ValueError(
            "Cannot derive a threshold from an empty population."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError(
            "uplift_scores must contain only finite values."
        )

    k = max(1, int(np.ceil(scores.size * capacity_fraction)))
    index = scores.size - k

    return float(np.partition(scores, index)[index])
