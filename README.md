# Project 2 — Production Uplift & Customer Decisioning API

A production-oriented ML decisioning system built around a **frozen V7.2 Logistic Regression T-learner**.

The project takes a validated customer-level uplift model and packages it as a real-time decision service with **FastAPI, Docker, automated testing, CI/CD, structured logging, Prometheus monitoring, explainability, cost-sensitive decisioning, and zero-cost public deployment**.

The productionisation layer deliberately does **not** retrain the model, reopen V7.9 Final Validation, or hard-code a supposedly optimal business policy.

---

## Live API

**Interactive Swagger UI**

https://project2-uplift-decision-api.onrender.com/docs

**Health**

https://project2-uplift-decision-api.onrender.com/health

**OpenAPI specification**

https://project2-uplift-decision-api.onrender.com/openapi.json

The public API is deployed as a Docker-based Render Web Service using the Free compute tier.

The deployment is intended for **portfolio demonstration and production-oriented engineering practice**, not for a production SLA.

---

## Problem

A conventional propensity model estimates:

> Who is likely to respond?

An uplift model asks a different question:

> Who is expected to respond **because of the treatment**?

For customer-level features `X` and treatment assignment `T`, the production score is:

```text
uplift =
P(Y=1 | X, T=1)
-
P(Y=1 | X, T=0)
```

The system therefore separates two concerns:

```text
Customer features
       ↓
Frozen ML model
       ↓
Control / Treatment predictions
       ↓
Uplift score
       ↓
Business decision policy
       ↓
TARGET / DO_NOT_TARGET
```

This separation allows the validated model to remain stable while business policies can evolve independently.

---

## Frozen Model

The production model is the validated **V7.2 Logistic Regression T-learner**.

| Component              | Production specification                    |
| ---------------------- | ------------------------------------------- |
| Model family           | Logistic Regression T-learner               |
| Model version          | `v7.2`                                      |
| Features               | `f0 ... f11`                                |
| Treatment field        | `treatment`                                 |
| Primary outcome        | `visit`                                     |
| Production score       | Raw uplift                                  |
| Preprocessing          | `StandardScaler` fitted on Development only |
| Production calibration | None                                        |
| Validation state       | V7.9 Final Validation frozen                |
| Alternative models     | RF/HGB rejected as primary uplift models    |

Productionisation does not modify the validated model.

No retraining, tuning, resampling, SMOTE, or class-weighting was introduced during this stage.

---

## Imbalanced Outcome & Evaluation

The primary outcome is `visit`, which is a relatively rare outcome in the Criteo dataset, with approximately **4.7% positive prevalence**.

Because this project is fundamentally an **uplift / treatment-effect problem**, it is not treated as an ordinary binary-classification optimisation problem.

Accuracy is therefore not used as the primary model-selection or business-decision criterion.

Evaluation instead prioritises evidence aligned with the actual decision problem, including:

* uplift and treatment-effect performance
* incremental outcome
* policy-level performance
* treatment-versus-control comparisons

The validated V7.2 model remains frozen during productionisation.

The project deliberately does not introduce generic imbalance interventions such as SMOTE, resampling, or class weighting merely to optimise conventional classification metrics.

See:

[`docs/MODEL_EVALUATION_AND_IMBALANCE_V8_0_1.md`](docs/MODEL_EVALUATION_AND_IMBALANCE_V8_0_1.md)

---

## Decisioning Architecture

Model inference and business policy are separate layers.

### Threshold Decisioning

`POST /decide`

Applies an explicitly supplied threshold to an uplift score.

```text
uplift_score
      +
threshold
      ↓
ThresholdPolicy
      ↓
TARGET / DO_NOT_TARGET
```

The threshold is treated as a configurable policy input rather than a universally optimal business rule.

### End-to-End Decisioning

`POST /decision`

Runs:

```text
customer features
      ↓
V7.2 inference
      ↓
uplift score
      ↓
threshold policy
      ↓
business decision
```

### Capacity Decisioning

The capacity primitive derives a threshold from a scored reference population for a requested targeting fraction.

This distinction is intentional:

> Top-X% targeting is a population-level concept. A single real-time record cannot determine a population percentile unless a threshold has already been derived from a reference population.

### Economic Decisioning

`POST /decision/economic`

Economic decisions require explicit runtime business inputs:

* treatment cost
* expected incremental benefit per unit uplift
* minimum acceptable net value

The decision is based on:

```text
net_value =
    uplift_score
    × expected_incremental_benefit_per_unit_uplift
    − treatment_cost
```

No treatment costs or business benefits are invented from the Criteo dataset.

Example:

```text
uplift_score = 0.05
benefit per unit uplift = 1.0
treatment cost = 0.01

net_value = 0.04
```

The system then compares `net_value` with `minimum_net_value`.

---

## API Surface

| Method | Endpoint             | Purpose                                    |
| ------ | -------------------- | ------------------------------------------ |
| GET    | `/health`            | Service health and active versions         |
| POST   | `/predict`           | Customer-level model inference             |
| POST   | `/explain`           | Model-level feature contributions          |
| POST   | `/decide`            | Threshold decision from an uplift score    |
| POST   | `/decision`          | End-to-end prediction + threshold decision |
| POST   | `/decision/economic` | Cost-sensitive economic decision           |
| GET    | `/metrics`           | Prometheus-compatible metrics              |

Every relevant response carries version identifiers such as:

```text
model_version
feature_schema_version
score_version
policy_version
```

This provides explicit traceability between model, schema, scoring logic, and business policy.

---

## Explainability

`POST /explain` exposes model-specific explanation information derived from the Logistic T-learner structure.

The response includes:

* control and treatment log-odds
* control and treatment intercepts
* raw and scaled feature values
* control feature contributions
* treatment feature contributions
* treatment-minus-control contributions

The explanation is generated from the same production model artifacts used for inference rather than from a separate surrogate model.

---

## Input Validation

The API uses explicit request schemas and rejects invalid inputs.

Examples include:

* missing features
* extra features
* non-numeric values
* NaN / infinity
* invalid uplift values
* invalid thresholds
* negative treatment costs
* missing economic inputs

Invalid requests return structured HTTP `422` validation responses.

This prevents silent coercion of malformed production requests.

---

## Artifact Integrity

The production system verifies model artifacts using **SHA-256 integrity information**.

This creates a clear separation between:

```text
validated model artifact
        +
runtime application
        +
business policy configuration
```

An unexpected artifact change can therefore be detected before an unknown model version is silently served.

---

## Observability

The service implements structured JSON logging.

Request-level operational metadata includes:

* `request_id`
* HTTP method
* endpoint
* status code
* latency
* model version
* policy version

ML events record relevant prediction and decision metadata without logging customer feature values.

### Prometheus Metrics

HTTP metrics:

```text
http_requests_total
http_request_duration_seconds
```

ML metrics:

```text
ml_predictions_total
ml_decisions_total
```

The public deployment has been manually validated through `/metrics`, confirming that HTTP traffic, model inference, and business decision events are being recorded.

---

## Testing

The project currently has:

```text
53 tests
53 passed
```

The test suite covers:

* API contracts
* feature validation
* prediction behaviour
* explainability consistency
* threshold decisioning
* capacity logic
* economic decisioning
* cost-sensitive boundary conditions
* invalid economic inputs
* model/service behaviour
* production API validation

---

## CI/CD

GitHub Actions provides continuous integration.

The CI workflow performs:

```text
Checkout
   ↓
Python setup
   ↓
Install test dependencies
   ↓
pytest
   ↓
Docker Buildx
   ↓
Docker image build
   ↓
Container startup
   ↓
HTTP smoke tests
```

The Docker smoke tests verify critical production paths including:

```text
/health
/predict
/decision
/decision/economic
```

The latest production documentation commit passed the complete CI workflow successfully.

---

## Docker

The application is packaged as a Docker image based on Python 3.10.

The container:

* installs the project from `pyproject.toml`
* includes source code, configuration, and model artifacts
* runs under a non-root `appuser`
* starts FastAPI with Uvicorn
* binds to `0.0.0.0`
* consumes the deployment platform's `PORT` environment variable
* falls back to port `8000` for local execution

The Docker image is therefore portable across local and cloud environments.

---

## Deployment

The public service is deployed as a Docker-based Render Web Service.

Current deployment configuration:

```text
Service type:      Web Service
Runtime:           Docker
Branch:            main
Compute:           Free
Health check:      /health
Persistent disk:   none
Secret files:      none
Manual env vars:   none
```

The public deployment has been validated through the actual production API.

Verified successfully:

```text
GET  /health                 → 200
POST /predict               → 200
POST /explain               → 200
POST /decide                → 200
POST /decision              → 200
POST /decision/economic     → 200
GET  /metrics               → Prometheus output
```

---

## Production Validation

The public Swagger interface was used to validate the deployed service end-to-end.

The deployed service reports:

```text
model_version           = v7.2
feature_schema_version  = features_v1
score_version           = raw_uplift_v1
policy_version          = v8.0.4-threshold-v1
```

The successful public requests also returned request identifiers, demonstrating end-to-end observability through the deployed API.

---

## Operational Limitations

The public deployment uses Render's Free compute tier.

This is intentionally a **zero-cost portfolio deployment**, not an enterprise production environment.

Known limitations include:

* service spin-down after inactivity
* cold-start latency
* limited CPU and memory
* finite free usage quotas
* ephemeral local filesystem
* no persistent disk

The deployment therefore demonstrates the engineering patterns required for production-oriented ML serving without claiming enterprise availability, autoscaling, or production SLA guarantees.

---

## Failure Handling & Rollback

The system is designed to fail explicitly rather than silently degrading.

### Model artifact failure

An unavailable or invalid model artifact should prevent reliable serving rather than silently falling back to an unknown model.

### Invalid request

Return HTTP `422` with structured validation information.

### Health check failure

The deployment platform can detect an unhealthy instance through `/health`.

### CI failure

A failed test or Docker smoke test means the change should not be considered release-ready.

### Deployment failure

The last known-good deployment remains the rollback target.

Rollback should include:

1. identify the last known-good commit
2. verify its CI result
3. identify the corresponding deployment
4. rollback
5. verify `/health`
6. verify model and policy versions
7. validate the critical decision endpoints

---

## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       └── docker-ci.yml
├── api/
│   ├── main.py
│   └── schemas.py
├── configs/
│   ├── model_v7_2.json
│   ├── policy.example.json
│   └── policy_v8_0_4.json
├── docs/
│   ├── MODEL_EVALUATION_AND_IMBALANCE_V8_0_1.md
│   ├── OPERATIONAL_MLOPS_V8_0_1.md
│   └── PRODUCTION_CONTRACT_V8_0_1.md
├── models/
│   └── v7.2/
├── src/
│   └── project2_core/
├── tests/
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## Local Development

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the project with test dependencies:

```powershell
pip install -e ".[test]"
```

Run the test suite:

```powershell
pytest -q
```

Start the API locally:

```powershell
uvicorn api.main:app --reload
```

Local Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Documentation

Additional technical documentation:

* [`Production Contract`](docs/PRODUCTION_CONTRACT_V8_0_1.md)
* [`Model Evaluation & Imbalance`](docs/MODEL_EVALUATION_AND_IMBALANCE_V8_0_1.md)
* [`Operational & MLOps Guide`](docs/OPERATIONAL_MLOPS_V8_0_1.md)

---

## Engineering Principles

The project follows several deliberate engineering principles:

1. **Freeze validated model behaviour before productionisation.**
2. **Separate model inference from business policy.**
3. **Make business assumptions explicit rather than embedding hidden assumptions in the model.**
4. **Validate every external input.**
5. **Track model, schema, score, and policy versions.**
6. **Verify production artifacts before serving them.**
7. **Observe both application and ML events.**
8. **Test the container, not only the Python functions.**
9. **Fail explicitly rather than silently degrade model or decision behaviour.**
10. **Treat the free-tier deployment as a portfolio demonstration rather than a production-SLA environment.**

---

## Current Status

The project currently demonstrates:

* production-oriented ML serving
* FastAPI and REST API design
* Docker containerisation
* automated testing
* CI/CD
* structured logging
* Prometheus monitoring
* model artifact integrity verification
* model and policy versioning
* model explainability
* imbalanced-outcome methodology
* threshold decisioning
* capacity policy primitives
* cost-sensitive economic decisioning
* public cloud deployment
* operational validation
