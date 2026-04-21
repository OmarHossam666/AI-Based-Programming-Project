import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data.pipeline import DataIngestionPipeline, CNNFeatureExtractor, feature_selection_pipeline
from models.cnn_model import BrainTumorCNN

def run_task2():
    print("="*50)
    print("TASK 2: DATA ENGINEERING & FEATURE SELECTION")
    print("="*50)

    # ---------------------------------------------------------
    # 1. Load MRI Data
    # ---------------------------------------------------------
    ingestion = DataIngestionPipeline(data_dir="data")
    # Assuming 'Training' is your subset folder
    X_raw, y = ingestion.load_data(subset="Training") 
    
    # ---------------------------------------------------------
    # 2. Extract 7200-dim CNN Features
    # ---------------------------------------------------------
    # Use the actual BrainTumorCNN model
    model = BrainTumorCNN() 
    
    extractor = CNNFeatureExtractor(
        model=model, 
        model_weights_path="best_model.pth"
    )
    
    tabular_X = extractor.extract(X_raw, batch_size=32)
    print(f"Extracted feature matrix shape: {tabular_X.shape}")

    # ---------------------------------------------------------
    # 3. Run Feature Selectors
    # ---------------------------------------------------------
    # Run Heuristic
    start_time = time.time()
    X_heur, mask_heur, score_heur, selector_heur = feature_selection_pipeline(
        tabular_X, y, method='heuristic', max_features=30
    )
    time_heur = time.time() - start_time

    # Run GA
    start_time = time.time()
    X_ga, mask_ga, score_ga, selector_ga = feature_selection_pipeline(
        tabular_X, y, method='ga', pop_size=15, generations=10, mutation_rate=0.01
    )
    time_ga = time.time() - start_time

    # ---------------------------------------------------------
    # 4. Compare and Print Results
    # ---------------------------------------------------------
    results_df = pd.DataFrame({
        'method': ['Heuristic', 'Genetic Algorithm'],
        'n_features_selected': [np.sum(mask_heur), np.sum(mask_ga)],
        'best_accuracy': [score_heur, score_ga],
        'time_seconds': [round(time_heur, 2), round(time_ga, 2)]
    })
    
    print("\n" + "="*50)
    print("FEATURE SELECTION COMPARISON REPORT")
    print("="*50)
    print(results_df.to_string(index=False))
    
    # ---------------------------------------------------------
    # 5. Plot Selection Curves Side-by-Side
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Heuristic
    h_steps = [h[0] for h in selector_heur.selection_history]
    h_scores = [h[2] for h in selector_heur.selection_history]
    axes[0].plot(h_steps, h_scores, marker='o', color='b', linewidth=2)
    axes[0].set_title('Heuristic: Accuracy vs Selected Features')
    axes[0].set_xlabel('Number of Features')
    axes[0].set_ylabel('Accuracy')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Genetic Algorithm
    ga_gens = range(1, len(selector_ga.best_score_per_generation_) + 1)
    axes[1].plot(ga_gens, selector_ga.best_score_per_generation_, marker='o', color='g', linewidth=2)
    axes[1].set_title('Genetic Algorithm: Convergence Curve')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Best Accuracy')
    axes[1].set_xticks(ga_gens)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('feature_selection_comparison.png')
    print("\nSaved comparison plot to 'feature_selection_comparison.png'")
    plt.show()

    # ---------------------------------------------------------
    # 6. Save Features as .npy files
    # ---------------------------------------------------------
    os.makedirs("saved_features", exist_ok=True)
    
    np.save('saved_features/X_heuristic.npy', X_heur)
    np.save('saved_features/mask_heuristic.npy', mask_heur)
    
    np.save('saved_features/X_ga.npy', X_ga)
    np.save('saved_features/mask_ga.npy', mask_ga)
    
    # Determine the overall best and save as canonical
    if score_heur >= score_ga:
        np.save('saved_features/X_selected.npy', X_heur)
        np.save('saved_features/mask_selected.npy', mask_heur)
        print("-> Heuristic performed best (or tied). Saved as canonical X_selected.npy")
    else:
        np.save('saved_features/X_selected.npy', X_ga)
        np.save('saved_features/mask_selected.npy', mask_ga)
        print("-> Genetic Algorithm performed best. Saved as canonical X_selected.npy")
        
    np.save('saved_features/y_labels.npy', y) # Save labels for Task 3 too!
    print("Data successfully staged for Task 3 (Ensemble Modeling).")

if __name__ == "__main__":
    run_task2()