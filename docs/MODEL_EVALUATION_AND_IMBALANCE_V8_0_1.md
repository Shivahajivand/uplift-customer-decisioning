```markdown
# Model Evaluation & Imbalance — V8.0.1

## 1. Outcome Imbalance

The primary modelling outcome is `visit`.

The Criteo dataset contains a relatively low positive prevalence for
this outcome (approximately 4.7%).

Therefore, the outcome is imbalanced and ordinary accuracy can be
misleading when used as a standalone measure of model quality.

---

## 2. Why Accuracy Is Not the Primary Criterion

This project is not designed as a conventional binary-classification
system whose primary objective is to maximise classification accuracy.

The production objective is uplift / treatment-effect modelling:

P(visit=1 | X, T=1)
-
P(visit=1 | X, T=0)

The resulting uplift score is then used for customer targeting
decisions.

For this reason, model evaluation prioritises evidence that is aligned
with the actual decision problem, including:

- uplift ranking performance
- incremental outcome
- policy value
- treatment-versus-control comparisons

Classification accuracy alone is therefore not used as the primary
model-selection or business-decision criterion.

---

## 3. Evaluation Principle

Outcome imbalance is considered at the interpretation and evaluation
level rather than by automatically introducing generic classification
techniques such as resampling or SMOTE.

The project does not assume that improving a conventional
classification metric necessarily improves uplift-based targeting
decisions.

The primary question remains:

> Does the model help identify customers who benefit incrementally from
> treatment?

---

## 4. Productionisation Decision

The validated V7.2 model is treated as a frozen production artifact.

During productionisation, no additional:

- resampling
- SMOTE
- class-weighting
- imbalance-driven retraining
- model retuning

was introduced solely to address the imbalanced outcome.

This preserves consistency with the previously validated model and
avoids introducing a new, unvalidated model variant during deployment.

---

## 5. Relationship to Decisioning

Imbalance-aware interpretation does not replace uplift-based decisioning.

The production decision flow remains:

Features
→ treatment/control predictions
→ uplift score
→ decision policy
→ TARGET / DO_NOT_TARGET

Economic and capacity decisions are therefore based on uplift and
business-policy inputs rather than on raw classification accuracy.

---

## 6. Scope

This document records the methodological decision taken during
productionisation.

It does not claim that the frozen V7.2 model is the globally optimal
classifier for the imbalanced `visit` outcome.

It documents why the production system deliberately preserves the
validated uplift model and evaluates it according to the decision
problem it was designed to solve.
```
