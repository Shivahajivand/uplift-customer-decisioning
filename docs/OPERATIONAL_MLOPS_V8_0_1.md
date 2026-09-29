# Project 2 — Operational & MLOps Guide — V8.0.1

## 1. Scope

This document describes the operational architecture and productionisation
of the frozen V7.2 Logistic T-learner.

The production system is designed as a portfolio-grade, production-oriented
ML service rather than a production-SLA system.

The validated V7.2 model remains frozen during productionisation.

No retraining, retuning, resampling, SMOTE, class-weighting, or reopening
of V7.9 Final Validation was introduced during deployment.

---

## 2. Architecture

The system consists of the following layers:

```text
Client
  |
  v
Public HTTPS
  |
  v
Render Web Service
  |
  v
FastAPI
  |
  +--------------------+
  |                    |
  v                    v
ModelService        Decision Layer
  |                    |
  v                    +--> ThresholdPolicy
V7.2 Logistic         +--> CapacityPolicy
T-learner             +--> Economic Decision
  |
  v
Prediction / Uplift
```

The production model is the frozen V7.2 Logistic Regression T-learner.

For each request, the model computes:

P(visit=1 | X, T=0)

and

P(visit=1 | X, T=1)

with production uplift defined as:

uplift = p_treatment - p_control

The decision layer is separate from model inference.

---

## 3. Model Lifecycle

### Development

The model was previously trained and validated during the development
and validation lifecycle.

### Validation

V7.9 Final Validation is treated as the final validated model state.

### Productionisation

The validated V7.2 artifact is packaged for production serving without
retraining or changing the validated model specification.

### Serving

The production service loads the frozen artifact at application startup.

### Version identifiers

Production responses expose:

* model_version
* feature_schema_version
* score_version
* policy_version

These identifiers make model and decision behaviour traceable across
requests and deployments.

---

## 4. Artifact Integrity

The production artifact is verified using SHA-256 integrity information.

The purpose is to detect unintended changes to model artifacts between
validation and serving.

The production package therefore distinguishes between:

* the validated model artifact
* the runtime application
* the decision policy configuration

This prevents accidental replacement of the validated model during
deployment.

---

## 5. API Contract

### GET /health

Returns service health and active version identifiers.

Used by the deployment platform for application-level health checking.

### POST /predict

Runs model inference for one customer.

Returns:

* p_control
* p_treatment
* uplift_score
* model_version
* feature_schema_version
* score_version

### POST /explain

Runs inference together with feature-level explanation information.

Returns:

* control and treatment log-odds
* model intercepts
* scaled feature values
* control contributions
* treatment contributions
* treatment-minus-control contributions

### POST /decide

Applies an externally supplied threshold to an already available uplift
score.

This keeps the threshold as a policy input rather than claiming that one
business threshold is universally optimal.

### POST /decision

Runs the full real-time flow:

customer features
→ model prediction
→ uplift score
→ threshold policy
→ business decision

### POST /decision/economic

Runs cost-sensitive decisioning using explicit runtime business inputs:

* treatment_cost
* expected_incremental_benefit_per_unit_uplift
* minimum_net_value

The economic decision is based on:

net_value =
uplift_score × expected_incremental_benefit_per_unit_uplift
− treatment_cost

No business cost or benefit assumptions are learned or invented from the
Criteo dataset.

### GET /metrics

Exposes Prometheus-compatible application and ML metrics.

---

## 6. Validation and Testing

The project uses automated testing across core logic and API behaviour.

The current test suite contains 53 passing tests.

Coverage includes:

* feature schema validation
* missing feature rejection
* extra feature rejection
* invalid type rejection
* NaN rejection
* model inference
* explainability consistency
* threshold policy
* capacity policy
* economic decisioning
* cost-sensitive boundary cases
* API validation
* structured response contracts

The Docker CI pipeline also performs a container smoke test against:

* /health
* /predict
* /decision
* /decision/economic

---

## 7. CI/CD

The repository uses GitHub Actions for continuous integration.

The CI workflow performs:

1. repository checkout
2. Python environment setup
3. installation of test dependencies
4. pytest execution
5. Docker Buildx setup
6. Docker image build
7. container startup
8. HTTP smoke testing

A deployment should only be considered ready when the CI workflow
completes successfully.

---

## 8. Containerisation

The service is packaged as a Docker image.

The container:

* uses Python 3.10
* installs the project from pyproject.toml
* includes the source code, configuration, and frozen model artifacts
* runs as a non-root user
* starts Uvicorn with FastAPI
* binds to 0.0.0.0
* consumes the runtime PORT environment variable

The Dockerfile uses a fallback port for local execution while remaining
compatible with cloud deployment.

---

## 9. Deployment

The service is deployed as a Docker-based Render Web Service.

Current deployment properties:

* service type: Web Service
* runtime: Docker
* branch: main
* compute plan: Free
* health check path: /health
* persistent disk: none
* secret files: none
* manually supplied environment variables: none

The public API URL is:

https://project2-uplift-decision-api.onrender.com

The interactive API documentation is available at:

https://project2-uplift-decision-api.onrender.com/docs

The OpenAPI specification is available at:

https://project2-uplift-decision-api.onrender.com/openapi.json

---

## 10. Monitoring and Observability

The service implements structured JSON logging.

Each completed request records operational metadata such as:

* request_id
* HTTP method
* endpoint
* status code
* latency
* model_version
* policy_version

ML events additionally record relevant model or decision metadata without
logging customer feature values.

Prometheus metrics include:

### HTTP metrics

* http_requests_total
* http_request_duration_seconds

### ML prediction metrics

* ml_predictions_total

### Business decision metrics

* ml_decisions_total

This allows operational monitoring to distinguish:

* API traffic
* model inference activity
* business decision activity
* request latency
* validation failures

---

## 11. Explainability

The production API exposes a model explainability endpoint.

For the Logistic T-learner, explanations are derived from the model's
linear structure and expose feature contributions for both treatment and
control models.

The explanation layer is reconciled with the model prediction rather
than being a separate approximate explanation system.

This makes the prediction path and explanation path traceable to the
same production artifact.

---

## 12. Decision Policies

The system intentionally separates model inference from business policy.

### Threshold policy

A supplied threshold is compared against uplift.

### Capacity policy

A population-level threshold can be derived from a reference scored
population for a requested targeting capacity.

The online system consumes the resulting threshold rather than attempting
to infer Top-X% from a single real-time record.

### Economic policy

The decision is based on explicit business inputs and expected net value.

No policy is labelled universally optimal.

---

## 13. Error Handling

The API rejects malformed or invalid inputs with HTTP 422.

Examples include:

* missing features
* extra features
* non-numeric features
* NaN values
* invalid uplift values
* invalid thresholds
* negative treatment cost
* missing economic inputs

This keeps the API contract explicit and prevents silent coercion of
invalid requests.

---

## 14. Failure Scenarios

### Model artifact failure

If the model artifact is unavailable or fails integrity validation,
application startup should fail rather than silently serving an unknown
model.

### Invalid request

Return HTTP 422 with structured validation information.

### Health check failure

The deployment platform can stop routing traffic to an unhealthy
instance and restart it according to its health-check behaviour.

### CI failure

A failed automated test or Docker smoke test blocks the release from
being considered production-ready.

### Deployment failure

Retain the last known-good deployment and roll back rather than replacing
a validated deployment with an unverified one.

### Free-tier cold start

The zero-cost hosting environment may spin down after inactivity.
Cold-start latency is therefore an expected operational limitation of
the portfolio deployment.

---

## 15. Rollback Strategy

Rollback should always target a previously validated deployment.

The rollback procedure is:

1. identify the last known-good commit
2. verify the associated CI result
3. identify the corresponding deployment
4. roll back to that deployment
5. verify /health
6. verify /predict
7. verify the relevant decision endpoint
8. review logs and metrics

The model version and policy version in the response provide additional
evidence that the expected deployment is active.

---

## 16. Production Validation

The current public deployment has been manually validated through the
public Swagger interface.

Verified successfully:

* GET /health → 200
* POST /predict → 200
* POST /explain → 200
* POST /decision → 200
* POST /decision/economic → 200
* GET /metrics → successful Prometheus output

The deployed service reports:

* model_version = v7.2
* feature_schema_version = features_v1
* score_version = raw_uplift_v1
* policy_version = v8.0.4-threshold-v1

Prometheus metrics also confirmed production traffic and ML decision
events.

---

## 17. Free Hosting Limitation

The Render deployment is intentionally zero-cost and is used for
portfolio demonstration and production-oriented engineering practice.

It should not be represented as a production-SLA service.

Expected limitations include:

* cold starts after inactivity
* limited CPU and memory
* finite free usage quotas
* ephemeral local filesystem
* no persistent disk on the selected free compute plan

The service therefore demonstrates deployment, observability,
containerisation, API serving, and MLOps practices without claiming
enterprise availability or production-scale capacity.

---

## 18. Operational Principles

The production system follows these principles:

1. Freeze validated model behaviour before deployment.
2. Keep model inference separate from business policy.
3. Make business assumptions explicit.
4. Validate every external input.
5. Track model and policy versions.
6. Make model artefacts reproducible and integrity-checkable.
7. Observe both application and ML events.
8. Test the container, not only the Python code.
9. Fail explicitly rather than silently degrading.
10. Treat zero-cost hosting as a portfolio deployment constraint, not a
    production-SLA guarantee.

---

## 19. Current Completion State

Project 2 currently demonstrates:

* production-oriented ML serving
* REST API design
* FastAPI
* Docker
* automated testing
* CI/CD
* structured logging
* Prometheus monitoring
* model versioning
* artifact integrity verification
* model explainability
* imbalanced-outcome methodology
* threshold decisioning
* capacity policy primitives
* cost-sensitive economic decisioning
* public cloud deployment
* operational validation

The project is considered feature-complete for the planned production
engineering scope.

Remaining work is limited to final documentation polish, repository
presentation, CV positioning, and interview preparation.

````
