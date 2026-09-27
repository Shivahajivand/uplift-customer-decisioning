import pytest
from project2_core.decision_engine import (
    CapacityPolicy,
    EconomicInputs,
    calculate_net_value,
    derive_capacity_threshold,
    economic_decision,
)


def test_invalid_capacity():
    with pytest.raises(ValueError):
        derive_capacity_threshold([0.1, 0.2], 0)


def test_population_threshold():
    assert derive_capacity_threshold(
        [0.1, 0.2, 0.3, 0.4],
        0.5,
    ) == pytest.approx(0.3)


def test_capacity_target():
    policy = CapacityPolicy(
        policy_version="v8.0.1",
        threshold=0.2,
    )

    decision, _ = policy.decide(0.25)

    assert decision == "TARGET"


def test_capacity_reject():
    policy = CapacityPolicy(
        policy_version="v8.0.1",
        threshold=0.2,
    )

    decision, _ = policy.decide(0.1)

    assert decision == "DO_NOT_TARGET"


def test_economic_inputs_are_explicit():
    inputs = EconomicInputs(
        treatment_cost=0.01,
        expected_incremental_benefit_per_unit_uplift=1.0,
    )

    decision, reason = economic_decision(
        0.05,
        inputs,
    )

    assert decision == "TARGET"
    assert "net_value=0.04000000" in reason


def test_economic_decision_rejects_negative_net_value():
    inputs = EconomicInputs(
        treatment_cost=0.06,
        expected_incremental_benefit_per_unit_uplift=1.0,
    )

    decision, reason = economic_decision(
        0.05,
        inputs,
    )

    assert decision == "DO_NOT_TARGET"
    assert "net_value=-0.01000000" in reason


def test_economic_decision_targets_at_minimum_net_value_boundary():
    inputs = EconomicInputs(
        treatment_cost=0.01,
        expected_incremental_benefit_per_unit_uplift=1.0,
        minimum_net_value=0.04,
    )

    decision, reason = economic_decision(
        0.05,
        inputs,
    )

    assert decision == "TARGET"
    assert "net_value=0.04000000" in reason


def test_economic_decision_respects_minimum_net_value():
    inputs = EconomicInputs(
        treatment_cost=0.01,
        expected_incremental_benefit_per_unit_uplift=1.0,
        minimum_net_value=0.05,
    )

    decision, reason = economic_decision(
        0.05,
        inputs,
    )

    assert decision == "DO_NOT_TARGET"
    assert "net_value=0.04000000" in reason
def test_calculate_net_value():
    inputs = EconomicInputs(
        treatment_cost=0.01,
        expected_incremental_benefit_per_unit_uplift=1.0,
    )

    assert calculate_net_value(
        0.05,
        inputs,
    ) == pytest.approx(0.04)


def test_economic_inputs_reject_negative_treatment_cost():
    inputs = EconomicInputs(
        treatment_cost=-0.01,
        expected_incremental_benefit_per_unit_uplift=1.0,
    )

    with pytest.raises(ValueError):
        calculate_net_value(
            0.05,
            inputs,
        )