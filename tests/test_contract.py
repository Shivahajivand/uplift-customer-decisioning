import numpy as np
import pytest
from project2_core.model_service import ModelService
from project2_core.schemas import FEATURES

def test_feature_schema():
    assert FEATURES == tuple(f"f{i}" for i in range(12))

def test_missing_feature():
    service = object.__new__(ModelService)
    with pytest.raises(ValueError):
        service._validate_features({f"f{i}": 0.0 for i in range(11)})

def test_nan():
    service = object.__new__(ModelService)
    payload = {f"f{i}": 0.0 for i in range(12)}
    payload["f3"] = np.nan
    with pytest.raises(ValueError):
        service._validate_features(payload)

def test_extra_feature():
    service = object.__new__(ModelService)
    payload = {f"f{i}": 0.0 for i in range(12)}
    payload["unexpected"] = 1.0
    with pytest.raises(ValueError):
        service._validate_features(payload)
