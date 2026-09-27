# PROJECT 2 — V8.0.1 Production Core

This milestone packages the frozen V7.2 Logistic T-learner for production-oriented serving without retraining, tuning, reopening V7.9 Final Validation, or hard-coding a supposedly optimal targeting policy.

## Frozen Model

* Model family: Logistic Regression T-learner
* Model version: `v7.2`
* Baseline features: `f0 ... f11`
* Treatment: `treatment`
* Primary outcome: `visit`
* Production score: `raw uplift = P(Y=1 | X,T=1) - P(Y=1 | X,T=0)`
* Preprocessing: `StandardScaler` fitted on Development only
* Calibration: none in production
* RF/HGB: rejected as primary uplift models
* V7.9 Final Validation: frozen

## Imbalanced Outcome & Evaluation

The primary outcome is `visit`, which is a relatively rare outcome (approximately 4.7% positive in the Criteo dataset).

Because this project is fundamentally an uplift / treatment-effect problem rather than a conventional binary-classification problem, accuracy is not used as the primary model-selection or decision metric.

Evaluation therefore prioritises uplift- and policy-level evidence, including incremental outcome and policy value, rather than optimising for classification accuracy alone.

The validated V7.2 model remains frozen during productionisation. No additional resampling, SMOTE, class-weighting, or imbalance-driven retraining was introduced at this stage.

## Decision Policy

The model ranks treatment-response heterogeneity. A separate policy layer converts that score into a business decision.

Supported core modes:

1. **Capacity:** target the top X% of a scored population, or use an explicitly supplied threshold derived from a reference population.
2. **Economic:** requires explicit business inputs such as treatment cost, expected incremental benefit, and minimum acceptable net value. No business costs or benefits are invented from the dataset.

No policy is declared universally optimal.

## Current Productionisation Status

Completed capabilities:

* Frozen V7.2 model artifact integration
* SHA-256 model artifact integrity verification
* FastAPI real-time inference
* `/predict` prediction endpoint
* `/explain` model explainability endpoint
* `/decide` threshold-based decision endpoint
* `/decision` end-to-end scoring and threshold decision endpoint
* `/decision/economic` cost-sensitive economic decision endpoint
* Docker containerisation
* Automated testing
* GitHub Actions CI
* Structured request and ML-event logging
* Prometheus application and ML metrics
* Model versioning and feature/score version identifiers
* Capacity and threshold decision primitives
* Imbalance-aware evaluation methodology
* Cost-sensitive decisioning with explicit business inputs

## Next Milestones

* Zero-cost deployment evaluation
* Final MLOps and operational documentation
* Final architecture review
* Final interview-readiness review
