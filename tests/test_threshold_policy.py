from project2_core.decision_engine import ThresholdPolicy


def test_threshold_policy_target():
    policy = ThresholdPolicy(
        policy_version="v8.0.3-threshold-v1",
        threshold=0.01,
    )

    decision, reason = policy.decide(0.05)

    assert decision == "TARGET"
    assert "threshold" in reason


def test_threshold_policy_do_not_target():
    policy = ThresholdPolicy(
        policy_version="v8.0.3-threshold-v1",
        threshold=0.01,
    )

    decision, reason = policy.decide(0.005)

    assert decision == "DO_NOT_TARGET"
    assert "threshold" in reason
