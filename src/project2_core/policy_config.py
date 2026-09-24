from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PolicyConfig:
    policy_version: str
    policy_type: str
    threshold_source: str


def load_policy_config(path: str | Path) -> PolicyConfig:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Policy configuration not found: {path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    required_fields = {
        "policy_version",
        "policy_type",
        "threshold_source",
    }

    missing = required_fields - data.keys()

    if missing:
        raise ValueError(
            f"Missing policy configuration fields: {sorted(missing)}"
        )

    return PolicyConfig(
        policy_version=str(data["policy_version"]),
        policy_type=str(data["policy_type"]),
        threshold_source=str(data["threshold_source"]),
    )
