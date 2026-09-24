import json

import pytest

from project2_core.policy_config import load_policy_config


def test_policy_config_loads():
    config = load_policy_config(
        "configs/policy_v8_0_4.json"
    )

    assert config.policy_version == "v8.0.4-threshold-v1"
    assert config.policy_type == "threshold"
    assert config.threshold_source == "runtime"


def test_policy_config_rejects_missing_file(tmp_path):
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_policy_config(missing)


def test_policy_config_rejects_missing_fields(tmp_path):
    invalid = tmp_path / "invalid.json"

    invalid.write_text(
        json.dumps({
            "policy_version": "v1"
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_policy_config(invalid)
