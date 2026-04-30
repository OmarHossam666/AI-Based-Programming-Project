import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SHAPExplainer:
    """
    Explainability module using SHAP (SHapley Additive exPlanations).
    This class demystifies the black-box ensemble model by calculating the
    exact contribution of each CNN-extracted feature to the final prediction.
    """

    def __init__(self, model, X_train):
        """
        Initializes the SHAP explainer.

        Args:
            model: The fitted sklearn model (e.g., the StackingEnsemble).
            X_train (np.ndarray): The training data used as the background dataset.
        """
        self.model = model
        self.X_train = X_train

        # Map feature indices back t o meaningful names for traceability
        self.feature_names = [f"CNN_feat_{i}" for i in range(X_train.shape[1])]

        # We need a background dataset to integrate over. Using 50 samples keeps
        # the KernelExplainer computationally feasible.
        self.background_data = shap.sample(self.X_train, 50)

        print("Initializing SHAP Explainer...")
        # Check if the model is a simple linear model (like standalone LR) or a complex ensemble
        if hasattr(model, "coef_") and not hasattr(model, "estimators_"):
            print("Detected Linear Model. Using Fast LinearExplainer.")
            self.explainer = shap.LinearExplainer(self.model, self.background_data)
        else:
            print(
                "Detected Complex Model/Ensemble. Using KernelExplainer (this may take a moment)."
            )
            # We use a wrapper function to predict the probability of the positive class (Tumor)
            # if predict_proba is available, otherwise we fall back to standard predict.
            if hasattr(self.model, "predict_proba"):
                predict_fn = lambda x: self.model.predict_proba(x)[:, 1]
            else:
                predict_fn = self.model.predict

            self.explainer = shap.KernelExplainer(predict_fn, self.background_data)

        self.shap_values = None

    def explain_global(self, X_eval, save_dir="saved_features"):
        """
        Generates global explanations (Bar and Beeswarm plots) for the top 20 features.

        Args:
            X_eval (np.ndarray): The validation dataset to evaluate.
            save_dir (str): Directory to save the generated plots.
        """
        print(f"Calculating global SHAP values for {X_eval.shape[0]} samples...")
        # To save time, we calculate SHAP values for a representative sample if the dataset is huge
        eval_sample = shap.sample(X_eval, min(X_eval.shape[0], 200))

        self.shap_values = self.explainer.shap_values(eval_sample)
        os.makedirs(save_dir, exist_ok=True)

        # 1. Summary Bar Plot (Magnitudes)
        plt.figure()
        shap.summary_plot(
            self.shap_values,
            eval_sample,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=20,
            show=False,
        )
        plt.title("SHAP Global Feature Importance (Top 20)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_summary_bar.png"))
        plt.close()

        # 2. Beeswarm Plot (Directional Impacts)
        plt.figure()
        shap.summary_plot(
            self.shap_values,
            eval_sample,
            feature_names=self.feature_names,
            max_display=20,
            show=False,
        )
        plt.title("SHAP Beeswarm Plot (Impact on Model Output)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "shap_beeswarm.png"))
        plt.close()

        print(f"Global SHAP plots saved to {save_dir}/")

    def explain_local(self, x_sample, label):
        """
        Generates a local explanation for a single MRI scan prediction.

        Args:
            x_sample (np.ndarray): A single sample of shape (1, n_features).
            label (int): The true label of the sample (0 for Healthy, 1 for Tumor).

        Returns:
            str: A human-readable caption explaining the prediction.
        """
        print("\nCalculating local SHAP values for the specific MRI scan...")

        # Calculate SHAP value for this specific instance
        shap_val = self.explainer.shap_values(x_sample)

        # Handle formats returned by different explainers
        if isinstance(shap_val, list):
            shap_val = shap_val[1]  # Take positive class if it's a list

        # Get the base value (expected value)
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

        # Create an Explanation object for the waterfall plot
        explanation = shap.Explanation(
            values=shap_val[0],
            base_values=base_value,
            data=x_sample[0],
            feature_names=self.feature_names,
        )

        # Plot Waterfall
        plt.figure()
        shap.waterfall_plot(explanation, max_display=10, show=True)

        # Generate Human-Readable Caption
        prediction = 1 if (base_value + np.sum(shap_val)) > 0.5 else 0
        pred_text = "Tumor Detected" if prediction == 1 else "Healthy"
        true_text = "Tumor" if label == 1 else "Healthy"

        # Find the top driving feature for this specific prediction
        top_feature_idx = np.argmax(np.abs(shap_val[0]))
        top_feature_name = self.feature_names[top_feature_idx]
        top_feature_impact = shap_val[0][top_feature_idx]
        direction = "increased" if top_feature_impact > 0 else "decreased"

        caption = (
            f"--- Local Explanation Summary ---\n"
            f"True Label: {true_text} | Model Prediction: {pred_text}\n"
            f"The feature that had the strongest influence on this specific prediction was '{top_feature_name}', "
            f"which {direction} the probability of detecting a tumor by {abs(top_feature_impact):.2%}.\n"
            f"---------------------------------"
        )
        print(caption)
        return caption

    def get_top_features(self, n=10):
        """
        Extracts the most globally important features based on mean absolute SHAP values.

        Args:
            n (int): Number of top features to return.

        Returns:
            pd.DataFrame: DataFrame containing feature names and their importance scores.
        """
        if self.shap_values is None:
            raise ValueError(
                "SHAP values not calculated yet. Run explain_global() first."
            )

        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)

        # Create a DataFrame
        importance_df = pd.DataFrame(
            {"Feature": self.feature_names, "Mean_Abs_SHAP": mean_abs_shap}
        )

        # Sort and get top n
        top_features = importance_df.sort_values(
            by="Mean_Abs_SHAP", ascending=False
        ).head(n)
        return top_features.reset_index(drop=True)
