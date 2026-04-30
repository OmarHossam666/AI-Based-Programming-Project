import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import joblib
import matplotlib.pyplot as plt
import os
import shap

# Import our custom modules
from models.cnn_model import BrainTumorCNN
from data.pipeline import CNNFeatureExtractor
from explainability.logic_rules import RuleBasedAssistant
from explainability.lime_explainer import LIMEExplainer
from explainability.shap_explainer import SHAPExplainer


# =====================================================================
# 2. Caching Models for Fast Inference
# =====================================================================
@st.cache_resource
def load_models():
    """Loads all required models into memory once to prevent reloading."""
    # 1. Load CNN Extractor
    cnn_model = BrainTumorCNN()
    extractor = CNNFeatureExtractor(
        model=cnn_model,
        model_weights_path="best_model.pth",
        target_layer_name="features",
    )

    # 2. Load Feature Mask and Training Data
    mask = np.load("saved_features/mask_selected.npy")
    X_train_selected = np.load("saved_features/X_selected.npy")
    y_train = np.load("saved_features/y_labels.npy")

    # 3. Load the Stacking Ensemble
    stacker = joblib.load("deployment_models/stacker_v1.pkl")

    # 4. Initialize Rule Assistant
    assistant = RuleBasedAssistant()
    # Identify the top 8 features for logical rules
    top_feature_indices = [
        np.where(mask == 1)[0][i] for i in range(min(8, np.sum(mask)))
    ]
    assistant.extract_thresholds(X_train_selected, y_train, list(range(len(top_feature_indices))))

    # 5. Initialize SHAP Explainer (Requires background data)
    shap_engine = SHAPExplainer(stacker, X_train_selected)

    return extractor, mask, stacker, assistant, shap_engine


# =====================================================================
# 3. Modern Streamlit UI Layout
# =====================================================================
def main():
    st.set_page_config(
        page_title="NeuroScan AI | Brain Tumor Diagnostic Assistant",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for Premium Look
    st.markdown(
        """
        <style>
        .main {
            background-color: #0e1117;
        }
        .stMetric {
            background-color: #1e2130;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #3e445e;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
        }
        h1 {
            color: #00d4ff;
            font-weight: 800;
        }
        .diagnosis-card {
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.title("🛡️ NeuroScan AI")
        st.info("Clinical Grade Brain Tumor Detection Pipeline using Stacking Ensembles & XAI.")
        st.divider()
        st.subheader("System Status")
        st.success("CNN Extractor: Online")
        st.success("Stacker Ensemble: Online")
        st.success("LIME/SHAP Modules: Ready")
        
        st.divider()
        st.markdown("### Guidance")
        st.write("1. Upload high-res MRI scan.")
        st.write("2. Review AI confidence.")
        st.write("3. Inspect LIME for spatial cues.")
        st.write("4. Verify with Logic rules.")

    # Main Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.title("🧠 Brain Tumor Diagnostic Assistant")
        st.subheader("Advanced Neuro-Symbolic AI Pipeline")
    
    # Load models
    extractor, mask, stacker, assistant, shap_engine = load_models()

    # File Uploader
    st.divider()
    uploaded_file = st.file_uploader(
        "📂 Drag and drop or browse for an MRI scan (JPG, PNG)", 
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        # UI Columns: Left for Input/Prediction, Right for Visual XAI
        col_input, col_viz = st.columns([1, 1.5])

        with col_input:
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            
            # Processing
            with st.spinner("Executing Full Inference Pipeline..."):
                # 1. Preprocess
                transform = transforms.Compose([
                    transforms.Resize((250, 250)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                img_tensor = transform(image).unsqueeze(0)

                # 2. Extract Features
                raw_features = extractor.extract(img_tensor.numpy())
                
                # 3. Mask & Predict
                selected_features = raw_features[:, mask == 1]
                prediction = stacker.predict(selected_features)[0]
                probs = stacker.predict_proba(selected_features)[0]
                confidence = probs[1] if prediction == 1 else probs[0]

                # 4. Logic Rules
                rule_result = assistant.apply_rules(selected_features)
                clinical_text = assistant.generate_explanation(prediction, confidence, rule_result)

            # Results Display
            st.subheader("Diagnostic Output")
            if prediction == 1:
                st.error(f"🚨 **TUMOR DETECTED**")
            else:
                st.success(f"✅ **HEALTHY / NO TUMOR**")
            
            st.metric("Confidence Score", f"{confidence:.2%}")
            
            st.markdown(f"**Clinical Reasoning:**")
            st.info(clinical_text)

        with col_viz:
            tab1, tab2, tab3 = st.tabs(["🖼️ LIME Overlay", "📊 SHAP Importance", "🧠 Rule Logic"])
            
            with tab1:
                st.markdown("#### Spatial Interpretability")
                st.write("Highlights the regions (superpixels) that influenced the prediction.")
                
                with st.spinner("Generating LIME superpixels..."):
                    img_np = np.array(image.resize((250, 250)))
                    lime_explainer = LIMEExplainer(extractor, mask, stacker)
                    explanation = lime_explainer.explain_instance(img_np, num_samples=500)
                    
                    from skimage.segmentation import mark_boundaries
                    temp, mask_lime = explanation.get_image_and_mask(
                        explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False
                    )
                    
                    fig_lime, ax_lime = plt.subplots(figsize=(8, 8))
                    ax_lime.imshow(mark_boundaries(temp / 255.0, mask_lime))
                    ax_lime.axis("off")
                    st.pyplot(fig_lime)
                    plt.close(fig_lime)

            with tab2:
                st.markdown("#### Feature Contribution (SHAP)")
                st.write("Waterfall plot showing how specific CNN features pushed the prediction.")
                
                with st.spinner("Calculating Shapley values..."):
                    # For waterfall, we need an Explanation object
                    # SHAPExplainer.explainer.shap_values gives the values
                    shap_val = shap_engine.explainer.shap_values(selected_features)
                    
                    # Handle multi-class output
                    if isinstance(shap_val, list):
                        shap_val = shap_val[1]
                    
                    fig_shap = plt.figure()
                    # We manually construct the explanation for the waterfall plot in the UI
                    base_val = shap_engine.explainer.expected_value
                    if isinstance(base_val, (list, np.ndarray)):
                        base_val = base_val[1] if len(base_val) > 1 else base_val[0]
                    
                    exp_obj = shap.Explanation(
                        values=shap_val[0],
                        base_values=base_val,
                        data=selected_features[0],
                        feature_names=[f"Feat_{i}" for i in range(selected_features.shape[1])]
                    )
                    shap.waterfall_plot(exp_obj, max_display=10, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()

            with tab3:
                st.markdown("#### Neuro-Symbolic Logic Rules")
                st.write("The following IF-THEN rules were evaluated against the selected feature set.")
                
                for rule in assistant.RULE_SET:
                    triggered = rule.condition(selected_features.flatten())
                    icon = "🚩" if triggered else "⚪"
                    status = "**Triggered**" if triggered else "Not Triggered"
                    st.markdown(f"{icon} **{rule.name}**: {rule.explanation} ({status})")

    else:
        # Welcome Screen / Empty State
        st.markdown(
            """
            <div style='text-align: center; padding: 50px;'>
                <h2 style='color: #3e445e;'>Ready for Analysis</h2>
                <p style='color: #666;'>Please upload a high-resolution MRI scan to begin the diagnostic pipeline.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning("Awaiting MRI Scan Upload...")

if __name__ == "__main__":
    main()
