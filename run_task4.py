import numpy as np
import pandas as pd
import os

from models.ensemble import BaseLearners, StackingEnsemble
from explainability.shap_explainer import SHAPExplainer
from explainability.logic_rules import RuleBasedAssistant


def find_sample_index(y_true, y_pred, target_true, target_pred):
    """Helper to find the first index matching a specific true/pred condition."""
    indices = np.where((y_true == target_true) & (y_pred == target_pred))[0]
    if len(indices) > 0:
        return indices[0]
    return None


def run_task4():
    print("=" * 50)
    print("TASK 4: EXPLAINABILITY & REASONING LAYER")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Load Data & Retrain Best Model (Stacking)
    # ---------------------------------------------------------
    try:
        X_best = np.load("saved_features/X_selected.npy")
        y = np.load("saved_features/y_labels.npy")
    except FileNotFoundError:
        print("Error: Could not find saved features. Please run run_task2.py first.")
        return

    print("Retraining Stacking Ensemble on canonical features...")
    learner_manager = BaseLearners()
    learner_manager.train_base_learners(X_best, y)

    stacker = StackingEnsemble()
    stacker.fit(learner_manager.X_train, learner_manager.y_train)
    model = stacker.stacker

    # Get predictions and probabilities for the validation set
    y_val_pred = model.predict(learner_manager.X_val)
    y_val_proba = model.predict_proba(learner_manager.X_val)[:, 1]
    y_val_true = learner_manager.y_val

    # ---------------------------------------------------------
    # 2. Instantiate SHAP & Run Global Explanations
    # ---------------------------------------------------------
    print("\n--- Phase 1: SHAP Global Explainability ---")
    shap_explainer = SHAPExplainer(model, learner_manager.X_train)
    shap_explainer.explain_global(learner_manager.X_val, save_dir="saved_features")

    # Extract the integer indices of the top 8 features from SHAP
    top_features_df = shap_explainer.get_top_features(n=8)
    top_feature_indices = [
        int(feat.split("_")[-1]) for feat in top_features_df["Feature"]
    ]
    print("\nTop SHAP Feature Indices:", top_feature_indices)

    # ---------------------------------------------------------
    # 3. Instantiate Rule-Based Assistant & Extract Thresholds
    # ---------------------------------------------------------
    print("\n--- Phase 2: Symbolic Logic Integration ---")
    assistant = RuleBasedAssistant()
    assistant.extract_thresholds(
        learner_manager.X_train, learner_manager.y_train, top_feature_indices
    )

    # ---------------------------------------------------------
    # 4. Pick 3 Test Samples (TP, TN, FN)
    # ---------------------------------------------------------
    print("\n--- Phase 3: Local Explanations & Rule Evaluation ---")

    tp_idx = find_sample_index(y_val_true, y_val_pred, 1, 1)  # True Positive
    tn_idx = find_sample_index(y_val_true, y_val_pred, 0, 0)  # True Negative
    fn_idx = find_sample_index(y_val_true, y_val_pred, 1, 0)  # False Negative

    # Fallback if the model is so good it has 0 False Negatives
    if fn_idx is None:
        print(
            "Wow! 100% Recall achieved. No False Negatives found. Using a False Positive instead."
        )
        fn_idx = find_sample_index(y_val_true, y_val_pred, 0, 1)
        fn_label_name = "False Positive (Healthy misclassified as Tumor)"
    else:
        fn_label_name = "False Negative (MISSED TUMOR - Clinically Critical)"

    test_cases = [
        {"name": "True Positive (Tumor correctly detected)", "idx": tp_idx},
        {"name": "True Negative (Healthy correctly detected)", "idx": tn_idx},
        {"name": fn_label_name, "idx": fn_idx},
    ]

    # ---------------------------------------------------------
    # 5 & 6. Generate Explanations and Save to Markdown
    # ---------------------------------------------------------
    md_content = "# Brain Tumor Detection: Explainability & Rule Set Catalogue\n\n"
    md_content += "This document provides human-readable explanations for the ML model's predictions by fusing SHAP local feature attribution with a symbolic IF-THEN reasoning layer.\n\n"

    for case in test_cases:
        idx = case["idx"]
        if idx is None:
            continue

        x_sample = learner_manager.X_val[idx : idx + 1]
        true_label = y_val_true[idx]
        pred_label = y_val_pred[idx]
        confidence = y_val_proba[idx] if pred_label == 1 else (1 - y_val_proba[idx])

        print(f"\nEvaluating Case: {case['name']}")

        # 1. Get Local SHAP explanation (will plot and print to console)
        local_caption = shap_explainer.explain_local(x_sample, true_label)

        # 2. Apply Symbolic Rules
        rule_result = assistant.apply_rules(x_sample)

        # 3. Generate Final Fused Explanation
        final_explanation = assistant.generate_explanation(
            pred_label, confidence, rule_result
        )

        print("\n=> FINAL CLINICAL EXPLANATION:")
        print(final_explanation)
        print("-" * 50)

        # Append to Markdown string
        md_content += f"## Case: {case['name']}\n"
        md_content += f"**True Label:** {'Tumor' if true_label == 1 else 'Healthy'} | **Model Prediction:** {'Tumor' if pred_label == 1 else 'Healthy'}\n\n"
        md_content += f"### Symbolic Logic Output\n"
        md_content += f"* **Triggered Rules:** {', '.join(rule_result.triggered_rules) if rule_result.triggered_rules else 'None'}\n"
        md_content += f"* **Estimated Risk Level:** {rule_result.risk_level}\n"
        md_content += f"* **Rule Evaluation:** {rule_result.explanation_text}\n\n"
        md_content += f"### Clinical Summary\n"
        md_content += f"> {final_explanation}\n\n"
        md_content += "---\n\n"

    # Write the markdown file
    md_filepath = "explanation_examples.md"
    with open(md_filepath, "w") as f:
        f.write(md_content)

    print(f"\nSuccess! Full explanation catalogue saved to '{md_filepath}'.")
    print("Task 4 Deliverables complete.")


if __name__ == "__main__":
    run_task4()
