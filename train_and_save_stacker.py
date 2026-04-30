import numpy as np
import joblib
import os
from models.ensemble import StackingEnsemble

def train_and_save():
    print("Loading staged features...")
    try:
        X = np.load("saved_features/X_selected.npy")
        y = np.load("saved_features/y_labels.npy")
    except FileNotFoundError:
        print("Error: saved_features/X_selected.npy or y_labels.npy not found.")
        return

    print(f"Training Stacking Ensemble on {X.shape[0]} samples and {X.shape[1]} features...")
    stacker_obj = StackingEnsemble()
    stacker_obj.fit(X, y)
    
    os.makedirs("deployment_models", exist_ok=True)
    save_path = "deployment_models/stacker_v1.pkl"
    joblib.dump(stacker_obj.stacker, save_path)
    print(f"Success! Stacking model saved to {save_path}")

if __name__ == "__main__":
    train_and_save()
