# Brain Tumor Detection: Explainability & Rule Set Catalogue

This document provides human-readable explanations for the ML model's predictions by fusing SHAP local feature attribution with a symbolic IF-THEN reasoning layer.

## Case: True Positive (Tumor correctly detected)
**True Label:** Tumor | **Model Prediction:** Tumor

### Symbolic Logic Output
* **Triggered Rules:** R1, R2, R3, R4, R5, R6
* **Estimated Risk Level:** High Risk
* **Rule Evaluation:** elevated activity in primary CNN features 12 and 3 (Core Tumor Pattern) and anomalous spatial activation in CNN feature 13 (Secondary Structural Irregularity) and co-activation of CNN features 0 and 12 (Asymmetric Border Profile) and irregular texture gradient detected in CNN feature 10 and widespread anomalous activation across 4 or more primary CNN spatial features and peripheral mass indicator present in CNN feature 14

### Clinical Summary
> Model predicts Tumor (confidence 86%). Rule R1, R2, R3, R4, R5, R6 triggered: elevated activity in primary CNN features 12 and 3 (Core Tumor Pattern) and anomalous spatial activation in CNN feature 13 (Secondary Structural Irregularity) and co-activation of CNN features 0 and 12 (Asymmetric Border Profile) and irregular texture gradient detected in CNN feature 10 and widespread anomalous activation across 4 or more primary CNN spatial features and peripheral mass indicator present in CNN feature 14 — consistent with tumour-associated spatial patterns.

---

## Case: True Negative (Healthy correctly detected)
**True Label:** Healthy | **Model Prediction:** Healthy

### Symbolic Logic Output
* **Triggered Rules:** R6
* **Estimated Risk Level:** Moderate Risk
* **Rule Evaluation:** peripheral mass indicator present in CNN feature 14

### Clinical Summary
> Model predicts Healthy (confidence 60%). Rule R6 triggered: peripheral mass indicator present in CNN feature 14 — consistent with tumour-associated spatial patterns.

---

## Case: False Negative (MISSED TUMOR - Clinically Critical)
**True Label:** Tumor | **Model Prediction:** Healthy

### Symbolic Logic Output
* **Triggered Rules:** R2, R4
* **Estimated Risk Level:** Moderate Risk
* **Rule Evaluation:** anomalous spatial activation in CNN feature 13 (Secondary Structural Irregularity) and irregular texture gradient detected in CNN feature 10

### Clinical Summary
> Model predicts Healthy (confidence 51%). Rule R2 and R4 triggered: anomalous spatial activation in CNN feature 13 (Secondary Structural Irregularity) and irregular texture gradient detected in CNN feature 10 — consistent with tumour-associated spatial patterns.

---

