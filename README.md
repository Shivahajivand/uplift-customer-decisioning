# PROJECT 2 — V8.0.1 Production Core

This milestone packages the frozen V7.2 Logistic T-learner for production-oriented
serving without retraining, tuning, reopening V7.9 Final Validation, or hard-coding
a supposedly optimal targeting policy.

## Frozen model
- Model family: Logistic Regression T-learner
- Model version: v7.2
- Baseline features: f0 ... f11
- Treatment: treatment
- Primary outcome: visit
- Production score: raw uplift = P(Y=1|X,T=1) - P(Y=1|X,T=0)
- Preprocessing: StandardScaler fitted on Development only
- Calibration: none in production
- RF/HGB: rejected as primary uplift models
- V7.9 Final Validation: frozen

## Decision policy
The model ranks treatment-response heterogeneity. A separate policy layer converts
that score into a business decision.

Supported core modes:
1. capacity: top X% in a scored population, or an explicitly supplied threshold
2. economic: requires explicit business inputs; no business costs/benefits are
   invented from the dataset

No policy is declared universally optimal.

## Next milestones
V8.0.2 artifact integration + FastAPI
V8.0.3 Docker
V8.0.4 expanded tests + CI
V8.0.5 logging/monitoring
V8.0.6 explainability
V8.0.7 zero-cost deployment evaluation
