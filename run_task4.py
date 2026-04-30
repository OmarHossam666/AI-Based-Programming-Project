import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from models.ensemble import BaseLearners, StackingEnsemble
from models.cnn_model import BrainTumorCNN
from data.pipeline import DataIngestionPipeline, CNNFeatureExtractor
from explainability.shap_explainer import SHAPExplainer
from explainability.logic_rules import RuleBasedAssistant
from explainability.lime_explainer import LIMEExplainer


def find_sample_index(y_true, y_pred, target_true, target_pred):
    """Helper to find the first index matching a specific true/pred condition."""
    indices = np.where((y_true == target_true) & (y_pred == target_pred))[0]
    if len(indices) > 0:
        return indices[0]
    return None


def run_task4():
    print("=" * 50)
    print("TASK 4: EXPLAINABILITY & REASONING LAYER (SHAP + LIME + LOGIC)")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Load Data & Synchronize Splits
    # ---------------------------------------------------------
    try:
        X_best = np.load("saved_features/X_selected.npy")
        y = np.load("saved_features/y_labels.npy")
        mask = np.load("saved_features/mask_selected.npy")
    except FileNotFoundError:
        print("Error: Could not find saved features. Please run run_task2.py first.")
        return

    # To run LIME, we need the raw images [250, 250, 3] for the validation set
    print("Loading raw images for LIME synchronization...")
    ingestion = DataIngestionPipeline(data_dir="data")
    X_raw, _ = ingestion.load_data(subset="Training")

    print("Retraining Stacking Ensemble and synchronizing splits...")
    learner_manager = BaseLearners()
    # BaseLearners uses random_state=42 for its split
    learner_manager.train_base_learners(X_best, y)

    # Split X_raw exactly like the tabular features were split
    from sklearn.model_selection import train_test_split
    _, X_raw_val, _, _ = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

    stacker = StackingEnsemble()
    stacker.fit(learner_manager.X_train, learner_manager.y_train)
    model = stacker.stacker

    # Get predictions for the validation set
    y_val_pred = model.predict(learner_manager.X_val)
    y_val_proba = model.predict_proba(learner_manager.X_val)[:, 1]
    y_val_true = learner_manager.y_val

    # ---------------------------------------------------------
    # 2. Initialize Explainers
    # ---------------------------------------------------------
    # 1. CNN Extractor (for LIME)
    cnn_model = BrainTumorCNN()
    extractor = CNNFeatureExtractor(
        model=cnn_model,
        model_weights_path="best_model.pth",
        target_layer_name="features"
    )

    # 2. SHAP
    shap_explainer = SHAPExplainer(model, learner_manager.X_train)
    shap_explainer.explain_global(learner_manager.X_val, save_dir="saved_features")

    # 3. LIME
    lime_explainer = LIMEExplainer(extractor, mask, model)

    # 4. Logic
    top_features_df = shap_explainer.get_top_features(n=8)
    top_feature_indices = [int(feat.split("_")[-1]) for feat in top_features_df["Feature"]]
    assistant = RuleBasedAssistant()
    assistant.extract_thresholds(learner_manager.X_train, learner_manager.y_train, top_feature_indices)

    # ---------------------------------------------------------
    # 3. Local Evaluation & Report Generation
    # ---------------------------------------------------------
    os.makedirs("saved_features/visuals", exist_ok=True)
    
    tp_idx = find_sample_index(y_val_true, y_val_pred, 1, 1)
    tn_idx = find_sample_index(y_val_true, y_val_pred, 0, 0)
    fn_idx = find_sample_index(y_val_true, y_val_pred, 1, 0)

    test_cases = [
        {"name": "True_Positive", "idx": tp_idx, "label": "Tumor correctly detected"},
        {"name": "True_Negative", "idx": tn_idx, "label": "Healthy correctly detected"},
        {"name": "False_Negative", "idx": fn_idx, "label": "Missed Tumor"},
    ]

    md_content = "# Brain Tumor Diagnostic Pipeline: Multi-Modal XAI Report\n\n"
    md_content += "This report fuses Spatial (LIME), Statistical (SHAP), and Symbolic (Logic) explainability.\n\n"

    for case in test_cases:
        idx = case["idx"]
        if idx is None: continue

        print(f"\nProcessing {case['name']}...")
        x_tabular = learner_manager.X_val[idx:idx+1]
        # X_raw_val images are [Channels, Height, Width] from pipeline.py. 
        # LIME needs [Height, Width, Channels] in [0, 255]
        x_raw_img = X_raw_val[idx]
        x_raw_img_hwc = (np.transpose(x_raw_img, (1, 2, 0)) * 255).astype(np.uint8)

        # 1. SHAP Logic
        local_caption = shap_explainer.explain_local(x_tabular, y_val_true[idx])
        rule_result = assistant.apply_rules(x_tabular)
        final_explanation = assistant.generate_explanation(y_val_pred[idx], y_val_proba[idx], rule_result)

        # 2. LIME Visualization
        save_path = f"saved_features/visuals/lime_{case['name']}.png"
        explanation = lime_explainer.explain_instance(x_raw_img_hwc, num_samples=500)
        lime_explainer.save_explanation(explanation, save_path)

        # Build Markdown
        md_content += f"## Case: {case['label']}\n"
        md_content += f"**Result:** {'Tumor' if y_val_pred[idx] == 1 else 'Healthy'} (Confidence: {y_val_proba[idx]:.1%})\n\n"
        md_content += f"![LIME Explanation]({save_path})\n\n"
        md_content += f"### Clinical Reasoning\n> {final_explanation}\n\n"
        md_content += f"* **Triggered Rules:** {', '.join(rule_result.triggered_rules) if rule_result.triggered_rules else 'None'}\n"
        md_content += "---\n\n"

    with open("explanation_report.md", "w") as f:
        f.write(md_content)

    print("\nSuccess! Multi-modal report saved to 'explanation_report.md'.")


if __name__ == "__main__":
    run_task4()
