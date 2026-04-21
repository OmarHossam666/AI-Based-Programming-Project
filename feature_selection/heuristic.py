import sys
import os

# Add the project root to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from data.pipeline import evaluate_features


class HeuristicSelector:
    """
    Greedy Forward Feature Selection (Heuristic Method).

    This selector starts with an empty feature set (all-zeros mask) and greedily
    adds the single feature that most improves the classification accuracy in each
    iteration. It halts when the maximum number of features is reached or when
    adding new features no longer improves the model's accuracy.
    """

    def __init__(self, classifier="knn"):
        """
        Args:
            classifier (str): The lightweight evaluator to use ('knn' or 'dt').
        """
        self.classifier = classifier
        self.selected_mask_ = None
        self.best_score_ = 0.0
        self.n_selected_ = 0
        self.selection_history = []

    def fit(self, X, y, max_features=50):
        """
        Fits the feature selector by greedily finding the best feature subset.

        Args:
            X (np.ndarray): The complete feature matrix (e.g., [N, 7200]).
            y (np.ndarray): The target labels.
            max_features (int): The maximum number of features to select.

        Returns:
            self
        """
        n_features = X.shape[1]

        # 1. Start with an all-zeros binary mask
        self.selected_mask_ = np.zeros(n_features, dtype=int)
        self.best_score_ = 0.0
        self.selection_history = []

        print(f"Starting Heuristic Feature Selection (max_features={max_features})...")

        # 2. Greedily add features
        for step in range(1, max_features + 1):
            best_feature = -1
            best_temp_score = self.best_score_

            # Evaluate adding each unused feature
            for i in range(n_features):
                if self.selected_mask_[i] == 1:
                    continue

                # Create a temporary mask to test this specific feature
                temp_mask = self.selected_mask_.copy()
                temp_mask[i] = 1

                # Evaluate the temp mask using the shared evaluation function
                score = evaluate_features(X, y, temp_mask, classifier=self.classifier)

                # Track the best performing feature of this iteration
                if score > best_temp_score:
                    best_temp_score = score
                    best_feature = i

            # 2. Stop early if no feature improves the score
            if best_feature == -1:
                print(
                    f"Stopping early at step {step-1}: No additional feature improved accuracy."
                )
                break

            # Permanently add the best feature to our mask
            self.selected_mask_[best_feature] = 1
            self.best_score_ = best_temp_score
            self.n_selected_ += 1

            # 3. Track selection history
            self.selection_history.append((step, best_feature, self.best_score_))

            print(
                f"Step {step}: Added Feature {best_feature} | Accuracy: {self.best_score_:.4f}"
            )

        print(
            f"\nSelection complete. Selected {self.n_selected_} features with final accuracy {self.best_score_:.4f}"
        )
        return self

    def transform(self, X):
        """
        Reduces the input X to only the selected features.

        Args:
            X (np.ndarray): The complete feature matrix.

        Returns:
            np.ndarray: The reduced feature matrix.
        """
        if self.selected_mask_ is None:
            raise ValueError("The selector has not been fitted yet. Call 'fit' first.")

        mask_bool = self.selected_mask_ == 1
        return X[:, mask_bool]

    def fit_transform(self, X, y, max_features=50):
        """
        Fits the selector and immediately returns the reduced feature matrix.
        """
        self.fit(X, y, max_features=max_features)
        return self.transform(X)

    def plot_selection_curve(self, save_path=None):
        """
        Plots the learning/selection curve showing Accuracy vs. Number of Selected Features.
        """
        if not self.selection_history:
            raise ValueError(
                "No history to plot. Ensure 'fit' has been called and features were selected."
            )

        # Extract steps and scores from history tuples
        steps = [h[0] for h in self.selection_history]
        scores = [h[2] for h in self.selection_history]

        plt.figure(figsize=(10, 6))
        plt.plot(
            steps,
            scores,
            marker="o",
            linestyle="-",
            color="b",
            linewidth=2,
            markersize=6,
        )
        plt.title(
            "Heuristic Feature Selection: Accuracy vs. Selected Features", fontsize=14
        )
        plt.xlabel("Number of Selected Features", fontsize=12)
        plt.ylabel("Validation Accuracy", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Selection curve saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Demo Setup: Create synthetic data to test the HeuristicSelector
    print("--- Heuristic Feature Selection Demo ---")
    
    # Generate synthetic features (e.g., 200 samples, 50 features)
    # Let's make features 5, 10, and 15 informative
    n_samples = 200
    n_features = 50
    X_synthetic = np.random.randn(n_samples, n_features)
    y_synthetic = np.random.randint(0, 2, n_samples)
    
    # Inject signal into features 5, 10, 15
    X_synthetic[:, 5] += 2 * y_synthetic
    X_synthetic[:, 10] += 1.5 * y_synthetic
    X_synthetic[:, 15] += 1.0 * y_synthetic

    # Initialize and Fit
    selector = HeuristicSelector(classifier="knn")
    selector.fit(X_synthetic, y_synthetic, max_features=10)
    
    # Plot results
    # selector.plot_selection_curve() # Uncomment to see the plot
    
    print("\nDemo completed.")

