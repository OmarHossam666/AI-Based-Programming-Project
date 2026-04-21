from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from models.evaluation import evaluate_model
from sklearn.ensemble import StackingClassifier


class BaseLearners:
    """
    Manages the initialization and training of diverse base classifiers for
    the Brain Tumor Ensemble Model.

    By utilizing heterogeneous algorithms, we ensure diverse prediction behaviors,
    allowing the ensemble to reduce individual model errors and produce a robust
    final prediction.
    """

    def __init__(self):
        self.models = {}
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def train_base_learners(self, X, y):
        """
        Performs a stratified 80/20 split and trains the base learners.

        Args:
            X (np.ndarray): The optimal selected feature matrix (e.g., loaded from X_selected.npy).
            y (np.ndarray): The corresponding binary labels.
        """
        print("Performing stratified 80/20 split (random_state=42)...")
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Training Data Shape: {self.X_train.shape}")
        print(f"Validation Data Shape: {self.X_val.shape}")

        # 1. Logistic Regression
        # Appropriate because: Provides a strong, globally balanced linear baseline
        # and feature weighting for the reduced CNN feature vectors.
        print("\nTraining Logistic Regression...")
        lr_model = LogisticRegression(max_iter=2000, random_state=42)
        lr_model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = lr_model

        # 2. Linear SVC
        # Appropriate because: Performs exceptionally well in high-dimensional feature spaces.
        # It focuses on margin maximization to ensure strict class separation between
        # tumorous and healthy visual descriptors.
        print("Training Linear SVC...")
        svm_model = LinearSVC(
            random_state=42, dual=False
        )  # dual=False recommended when n_samples > n_features
        svm_model.fit(self.X_train, self.y_train)
        self.models["Linear SVM"] = svm_model

        # 3. Gaussian Naive Bayes
        # Appropriate because: Leverages statistical evidence from continuous feature
        # distributions. It assumes feature independence, offering a completely different
        # inductive bias compared to margin or regression-based models.
        print("Training Gaussian Naive Bayes...")
        nb_model = GaussianNB()
        nb_model.fit(self.X_train, self.y_train)
        self.models["Naive Bayes"] = nb_model

        print("\nAll base learners trained successfully.")
        return self.models


class VotingEnsemble:
    """
    Implements Voting Ensembles to combine the predictions of the base learners.
    Supports both Hard Voting (majority rule on class labels) and Soft Voting
    (averaging class probabilities).
    """

    def __init__(self):
        self.hard_voter = None
        self.soft_voter = None

    def fit(self, X_train, y_train):
        """
        Initializes and trains both hard and soft voting ensembles.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        # 1. Base models for Hard Voting
        lr = LogisticRegression(max_iter=2000, random_state=42)
        svm_hard = LinearSVC(random_state=42, dual=False)
        nb = GaussianNB()

        self.hard_voter = VotingClassifier(
            estimators=[("lr", lr), ("svm", svm_hard), ("nb", nb)], voting="hard"
        )

        # 2. Base models for Soft Voting
        # Note: LinearSVC does not support predict_proba(). We must swap it out for
        # SVC(kernel='linear', probability=True) to allow the soft voter to access probabilities.
        svm_soft = SVC(
            kernel="linear", probability=True, max_iter=2000, random_state=42
        )

        self.soft_voter = VotingClassifier(
            estimators=[("lr", lr), ("svm", svm_soft), ("nb", nb)], voting="soft"
        )

        print("Training Hard Voting Ensemble...")
        self.hard_voter.fit(X_train, y_train)

        print("Training Soft Voting Ensemble...")
        self.soft_voter.fit(X_train, y_train)

        return self

    def evaluate(self, X_val, y_val):
        """
        Evaluates both voting ensembles on the validation set.

        Returns:
            dict: A dictionary containing the evaluation metrics for both models.
        """
        # TRADE-OFF IN MEDICAL CLASSIFICATION:
        # Hard voting treats all models' final binary decisions equally (majority rules).
        # Soft voting averages the predicted probabilities, giving more weight to highly confident models.
        # In medical tasks like brain tumor detection, soft voting is often preferred because probability
        # calibration matters heavily—a base model that is 99% confident about a tumor presence should
        # naturally carry more weight than a model that is only 51% confident.

        print("Evaluating Voting Ensembles...")
        hard_metrics = evaluate_model(self.hard_voter, X_val, y_val)
        soft_metrics = evaluate_model(self.soft_voter, X_val, y_val)

        return {"Voting (Hard)": hard_metrics, "Voting (Soft)": soft_metrics}


class StackingEnsemble:
    """
    Implements a Stacking Ensemble that learns how to optimally combine the
    predictions of base learners using a meta-learner.
    """

    def __init__(self):
        self.stacker = None

    def explain_architecture(self):
        """
        Prints a clear description of the Stacking architecture and the role of
        cross-validation based on the Lab 3 theoretical concepts.
        """
        explanation = """
        === Stacking Ensemble Architecture ===
        1. Level 0 (Base Models): 
           Multiple diverse models (Logistic Regression, SVC, GaussianNB) are trained 
           on the selected features. Each captures different patterns in the MRI data.
           
        2. Level 1 (Meta-Learner): 
           A Logistic Regression model is trained using ONLY the predictions of the 
           Level 0 models as its input features (meta-features). It learns when to 
           trust each base model.
           
        3. Why Cross-Validation (cv=5)?
           To avoid overfitting, base models generate predictions on unseen folds 
           (out-of-fold predictions) during training. These out-of-fold predictions 
           are what the meta-learner trains on, ensuring it learns generalizable 
           relationships rather than just memorizing the training data.
        ======================================
        """
        print(explanation)

    def fit(self, X_train, y_train):
        """
        Initializes and trains the stacking ensemble, then extracts and displays
        the meta-learner's learned coefficients.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
        """
        # Define Level-0 estimators
        base_learners = [
            ("lr", LogisticRegression(max_iter=2000, random_state=42)),
            (
                "svm",
                SVC(kernel="linear", probability=True, max_iter=2000, random_state=42),
            ),
            ("nb", GaussianNB()),
        ]

        # Define Level-1 meta-learner
        meta_learner = LogisticRegression(max_iter=2000, random_state=42)

        # Initialize StackingClassifier
        # passthrough=False ensures the meta-learner only sees the predictions
        # from the base learners, not the original 250 GA-selected features.
        self.stacker = StackingClassifier(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=5,
            passthrough=False,
        )

        print("Training Stacking Ensemble (this may take a moment due to 5-fold CV)...")
        self.stacker.fit(X_train, y_train)

        # Extract and display Meta-Learner Coefficients
        # The coefficients tell us how much weight the meta-learner assigns to each base model
        print("\n--- Meta-Learner Learned Weights ---")
        coefs = self.stacker.final_estimator_.coef_[0]
        for name, coef in zip(
            ["Logistic Regression", "Linear SVM", "Naive Bayes"], coefs
        ):
            print(f"{name:>20}: {coef:+.4f}")

        print(
            "\nInterpretation: Higher positive values mean the stacker strongly trusts"
        )
        print("that model's prediction for classifying a Tumor (Class 1).")

        return self

    def evaluate(self, X_val, y_val):
        """
        Evaluates the stacking ensemble on the validation set.

        Returns:
            dict: The evaluation metrics.
        """
        print("\nEvaluating Stacking Ensemble...")
        metrics = evaluate_model(self.stacker, X_val, y_val)
        return {"Stacking": metrics}
