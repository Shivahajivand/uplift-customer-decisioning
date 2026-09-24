# Production Contract — V8.0.1

## Input
Exactly 12 numeric, finite features are required: f0 ... f11.

Reject:
- missing features
- extra features
- non-numeric values
- NaN / +/- infinity

The API should not require treatment as an arbitrary user-selected model input.
The T-learner computes both potential-outcome predictions internally.

## Prediction
p_control = P(visit=1 | X, T=0)
p_treatment = P(visit=1 | X, T=1)
uplift = p_treatment - p_control

Every response must carry model and schema version identifiers.

## Decision
Decision is a separate layer.

### Capacity
Capacity is population-based. In batch mode, sort the scored population and
derive the threshold corresponding to the requested capacity fraction.

For one real-time record, Top-X% is undefined unless a threshold has already been
derived from a reference population. Therefore the online primitive is threshold-
based.

### Economic
Economic decisions require explicit business inputs such as treatment cost and
expected incremental benefit. These are configurable inputs, not values learned
from the Criteo dataset.

## Versioning
Track:
- model_version
- policy_version
- feature_schema_version
- score_version

## Non-goals
This milestone does not retrain, tune, alter V7.9, choose a new winning policy,
or invent economic assumptions.
