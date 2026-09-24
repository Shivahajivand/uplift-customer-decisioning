import pytest
from project2_core.decision_engine import (
    CapacityPolicy, EconomicInputs, derive_capacity_threshold, economic_decision
)

def test_invalid_capacity():
    with pytest.raises(ValueError):
        derive_capacity_threshold([0.1, 0.2], 0)

def test_population_threshold():
    assert derive_capacity_threshold([0.1, 0.2, 0.3, 0.4], 0.5) == pytest.approx(0.3)

def test_capacity_target():
    policy = CapacityPolicy(policy_version="v8.0.1", threshold=0.2)
    decision, _ = policy.decide(0.25)
    assert decision == "TARGET"

def test_capacity_reject():
    policy = CapacityPolicy(policy_version="v8.0.1", threshold=0.2)
    decision, _ = policy.decide(0.1)
    assert decision == "DO_NOT_TARGET"

def test_economic_inputs_are_explicit():
    inputs = EconomicInputs(0.01, 1.0)
    decision, _ = economic_decision(0.05, inputs)
    assert decision == "TARGET"
