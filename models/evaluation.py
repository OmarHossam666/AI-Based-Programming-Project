import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def evaluate_model(model, X, y):
    """
    Evaluates a trained binary classifier and returns standard metrics.
    Uses 'binary' averaging specifically for the 2-class tumor task.
    """
    y_pred = model.predict(X)

    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred, average="binary", zero_division=0),
        "Recall": recall_score(y, y_pred, average="binary", zero_division=0),
        "F1-score": f1_score(y, y_pred, average="binary", zero_division=0),
    }


def compare_models(models_dict, X, y):
    """
    Evaluates multiple trained models and returns a styled comparison table.

    Args:
        models_dict (dict): Dictionary in the form {"Model Name": trained_model}
        X (np.ndarray): Validation feature matrix.
        y (np.ndarray): Validation labels.

    Returns:
        pd.io.formats.style.Styler: A styled Pandas DataFrame highlighting the best scores.
    """
    results = []

    for model_name, model in models_dict.items():
        metrics = evaluate_model(model, X, y)
        results.append(
            {
                "Model": model_name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1-score": metrics["F1-score"],
            }
        )

    df = pd.DataFrame(results).set_index("Model")

    # Apply styling to highlight the maximum value in each column (metric)
    styled_df = df.style.highlight_max(
        axis=0, props="background-color: lightgreen; color: black;"
    ).format("{:.4f}")
    return styled_df


def plot_confusion_matrix(model, X, y, model_name):
    """
    Generates and plots a seaborn heatmap for the confusion matrix.
    Labels are mapped to 'Healthy' (0) and 'Tumor' (1).
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Tumor"],
        yticklabels=["Healthy", "Tumor"],
    )

    plt.title(f"Confusion Matrix: {model_name}", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Unit Testing Module (Lab 4)
# =====================================================================


def test_prediction_shape(model, X):
    """Asserts that the prediction output length matches the input size."""
    y_pred = model.predict(X)
    assert len(y_pred) == len(
        X
    ), f"Prediction length ({len(y_pred)}) does not match input size ({len(X)})"


def test_metric_range(metrics_dict):
    """Asserts that all evaluation metric values fall within the valid [0, 1] range."""
    for metric_name, value in metrics_dict.items():
        assert (
            0.0 <= value <= 1.0
        ), f"{metric_name} is out of the valid range [0, 1]: {value}"


def test_evaluation_output_structure(metrics_dict):
    """Asserts that the required metric keys exist in the output dictionary."""
    required_keys = {"Accuracy", "Precision", "Recall", "F1-score"}
    assert required_keys.issubset(
        metrics_dict.keys()
    ), f"Missing required metrics. Found: {metrics_dict.keys()}"


def run_all_tests(models_dict, X, y):
    """
    Runs all basic unit tests on every model in the pipeline.
    """
    print("\n=== Running Unit Tests ===")
    all_passed = True
    for model_name, model in models_dict.items():
        try:
            # Test shape
            test_prediction_shape(model, X)

            # Generate metrics and test them
            metrics = evaluate_model(model, X, y)
            test_metric_range(metrics)
            test_evaluation_output_structure(metrics)

            print(f"[PASS] {model_name}: All tests passed.")
        except AssertionError as e:
            print(f"[FAIL] {model_name}: {e}")
            all_passed = False

    if all_passed:
        print(
            "-> Pipeline Reliability Verified: All basic unit tests passed successfully."
        )
    return all_passed


# =====================================================================
# Experiment Tracking Module (Lab 4)
# =====================================================================


class ExperimentTracker:
    """
    Tracks and logs model configurations and evaluation results to ensure reproducibility.
    """

    def __init__(self):
        self.experiment_log = pd.DataFrame(
            columns=[
                "Model",
                "Dataset",
                "Num_Features",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
            ]
        )

    def log_experiment(self, model_name, dataset, n_features, metrics_dict):
        """Logs a single model's experiment results into the tracker."""
        self.experiment_log.loc[len(self.experiment_log)] = [
            model_name,
            dataset,
            n_features,
            metrics_dict.get("Accuracy"),
            metrics_dict.get("Precision"),
            metrics_dict.get("Recall"),
            metrics_dict.get("F1-score"),
        ]

    def save(self, path="experiment_log.csv"):
        """Saves the recorded experiment logs to a CSV file."""
        self.experiment_log.to_csv(path, index=False)
        print(f"Experiment results saved successfully to {path}")
