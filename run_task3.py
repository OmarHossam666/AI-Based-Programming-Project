import numpy as np
import os
import pandas as pd
from models.ensemble import BaseLearners, VotingEnsemble, StackingEnsemble
from models.evaluation import (
    compare_models,
    evaluate_model,
    run_all_tests,
    ExperimentTracker,
)


def run_task3():
    print("=" * 50)
    print("TASK 3: ENSEMBLE MODELING & EVALUATION")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Load Best Selected Features (Canonical)
    # ---------------------------------------------------------
    try:
        # Load the canonical best features determined by Task 2
        X_best = np.load("saved_features/X_selected.npy")
        y = np.load("saved_features/y_labels.npy")
    except FileNotFoundError:
        print("Error: Could not find saved features. Please run run_task2.py first.")
        return

    n_features = X_best.shape[1]
    dataset_name = "Brain Tumor MRI"
    print(
        f"Loaded optimal selected features: {n_features} features for {X_best.shape[0]} samples."
    )

    # ---------------------------------------------------------
    # 2. Initialize and Train Models
    # ---------------------------------------------------------
    print("\n--- Phase 1: Base Learners ---")
    learner_manager = BaseLearners()
    base_models = learner_manager.train_base_learners(X_best, y)

    # We use the same train/val splits managed by BaseLearners to ensure fairness
    X_train, X_val = learner_manager.X_train, learner_manager.X_val
    y_train, y_val = learner_manager.y_train, learner_manager.y_val

    print("\n--- Phase 2: Voting Ensembles ---")
    voting_ensembles = VotingEnsemble()
    voting_ensembles.fit(X_train, y_train)

    print("\n--- Phase 3: Stacking Ensemble ---")
    stacker = StackingEnsemble()
    stacker.explain_architecture()
    stacker.fit(X_train, y_train)

    # Compile all models into a single dictionary for testing and tracking
    all_models = {**base_models}
    all_models["Voting (Hard)"] = voting_ensembles.hard_voter
    all_models["Voting (Soft)"] = voting_ensembles.soft_voter
    all_models["Stacking"] = stacker.stacker

    # ---------------------------------------------------------
    # 3. Unit Testing
    # ---------------------------------------------------------
    # Runs the shape, range, and structure assertions
    run_all_tests(all_models, X_val, y_val)

    # ---------------------------------------------------------
    # 4. Experiment Tracking
    # ---------------------------------------------------------
    print("\n=== Tracking Experiments ===")
    tracker = ExperimentTracker()

    for name, model in all_models.items():
        metrics = evaluate_model(model, X_val, y_val)
        tracker.log_experiment(name, dataset_name, n_features, metrics)

    tracker.save("experiment_log.csv")

    # ---------------------------------------------------------
    # 5. Final Comparison
    # ---------------------------------------------------------
    print("\n=== Final Model Comparison Table ===")
    comparison_table = compare_models(all_models, X_val, y_val)

    # If running in a script (not a Jupyter notebook), .style returns a Styler object
    # which doesn't print well. We will extract and print the raw dataframe.
    if hasattr(comparison_table, "data"):
        print(comparison_table.data.to_string())
    else:
        print(comparison_table.to_string())


if __name__ == "__main__":
    run_task3()
