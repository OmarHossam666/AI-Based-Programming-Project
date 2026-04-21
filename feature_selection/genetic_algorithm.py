import sys
import os

# Add the project root to sys.path to allow running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import matplotlib.pyplot as plt
from data.pipeline import evaluate_features


class GeneticSelector:
    """
    Genetic Algorithm Feature Selection.

    This class mimics biological evolution to find an optimal feature subset based on Lab 2 theory:
    - Chromosome / Individual: A binary string where 1 means the feature is selected, 0 means ignored.
    - Population: A collection of multiple feature subsets (chromosomes).
    - Fitness Function: The accuracy of a classifier (e.g., KNN) trained on the selected features.
    - Selection (Elitism): The best-performing individuals are chosen as parents for the next generation.
    - Crossover: Two parent chromosomes swap segments at a random point to create offspring.
    - Mutation: Randomly flipping a 0 to 1 or 1 to 0 in offspring to maintain diversity and avoid local optima.
    """

    def __init__(self, classifier="knn"):
        """
        Args:
            classifier (str): The lightweight evaluator to use ('knn' or 'dt').
        """
        self.classifier = classifier
        self.selected_mask_ = None
        self.best_individual_ = None
        self.best_score_ = 0.0
        self.best_score_per_generation_ = []

    def initialize_population(self, pop_size, n_features):
        """Creates the initial generation zero randomly."""
        population = []
        for _ in range(pop_size):
            individual = np.random.randint(0, 2, n_features)
            population.append(individual)
        return np.array(population)

    def select_parents(self, population, fitness_scores, num_parents):
        """Selects the top-performing individuals based on fitness scores (argsort elitism)."""
        # argsort sorts ascending, so we take the last `num_parents` elements
        parents_idx = np.argsort(fitness_scores)[-num_parents:]
        return population[parents_idx]

    def crossover(self, p1, p2):
        """Combines two parents using single-point crossover."""
        # Ensure we don't split at the very beginning or very end
        point = random.randint(1, len(p1) - 1)
        child = np.concatenate([p1[:point], p2[point:]])
        return child

    def mutate(self, individual, mutation_rate=0.01):
        """Randomly flips bits in the individual based on the mutation rate."""
        for i in range(len(individual)):
            if random.random() < mutation_rate:
                individual[i] = 1 - individual[i]  # Flips 0 to 1, or 1 to 0
        return individual

    def fit(self, X, y, pop_size=20, generations=15, mutation_rate=0.01):
        """
        Runs the Genetic Algorithm to find the best feature subset.
        """
        n_features = X.shape[1]
        population = self.initialize_population(pop_size, n_features)

        self.best_score_per_generation_ = []
        self.best_score_ = 0.0
        self.best_individual_ = None

        epochs_without_improvement = 0

        print(
            f"Starting Genetic Algorithm (Pop: {pop_size}, Gen: {generations}, MutRate: {mutation_rate})..."
        )

        for generation in range(generations):
            # 1. Evaluate fitness for all individuals in current population
            fitness_scores = []
            for individual in population:
                score = evaluate_features(X, y, individual, classifier=self.classifier)
                fitness_scores.append(score)

            fitness_scores = np.array(fitness_scores)

            # 2. Track the best score of this generation
            max_idx = np.argmax(fitness_scores)
            gen_best_score = fitness_scores[max_idx]
            gen_best_individual = population[max_idx]

            self.best_score_per_generation_.append(gen_best_score)

            print(f"Generation {generation + 1} | Best Accuracy: {gen_best_score:.4f}")

            # 3. Update overall best and check early stopping
            if gen_best_score > self.best_score_:
                self.best_score_ = gen_best_score
                self.best_individual_ = gen_best_individual.copy()
                self.selected_mask_ = self.best_individual_.copy()
                epochs_without_improvement = 0  # Reset counter
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 3:
                print(
                    f"Early stopping triggered at Generation {generation + 1}: No improvement for 3 consecutive generations."
                )
                break

            # If we are at the last generation, no need to create a new one
            if generation == generations - 1:
                break

            # 4. Create Next Generation
            # Select parents (keeping top half of the population)
            num_parents = max(2, pop_size // 2)
            parents = self.select_parents(population, fitness_scores, num_parents)

            new_population = []

            # Fill the new population
            while len(new_population) < pop_size:
                # Randomly pick two parents from the elite pool
                p1, p2 = random.sample(list(parents), 2)

                # Crossover
                child = self.crossover(p1, p2)

                # Mutation
                child = self.mutate(child, mutation_rate)

                new_population.append(child)

            population = np.array(new_population)

        print(f"\nGA Complete. Best Overall Accuracy: {self.best_score_:.4f}")
        print(f"Selected Features: {np.sum(self.selected_mask_)} out of {n_features}")
        return self

    def transform(self, X):
        """Reduces the input X to only the features selected by the best GA chromosome."""
        if self.selected_mask_ is None:
            raise ValueError("The selector has not been fitted yet. Call 'fit' first.")

        mask_bool = self.selected_mask_ == 1
        return X[:, mask_bool]

    def fit_transform(self, X, y, pop_size=20, generations=15, mutation_rate=0.01):
        """Fits the GA selector and immediately returns the reduced feature matrix."""
        self.fit(
            X,
            y,
            pop_size=pop_size,
            generations=generations,
            mutation_rate=mutation_rate,
        )
        return self.transform(X)

    def plot_convergence(self, save_path=None):
        """Plots the GA convergence curve showing Best Accuracy vs. Generation."""
        if not self.best_score_per_generation_:
            raise ValueError("No history to plot. Ensure 'fit' has been called.")

        generations = range(1, len(self.best_score_per_generation_) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(
            generations,
            self.best_score_per_generation_,
            marker="o",
            linestyle="-",
            color="g",
            linewidth=2,
            markersize=6,
        )
        plt.title("Genetic Algorithm: Convergence Curve", fontsize=14)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Best Validation Accuracy", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(generations)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Convergence curve saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Demo Setup: Create synthetic data to test the GeneticSelector
    print("--- Genetic Algorithm Feature Selection Demo ---")

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
    selector = GeneticSelector(classifier="knn")
    selector.fit(X_synthetic, y_synthetic, pop_size=20, generations=15, mutation_rate=0.01)

    # Plot results
    # selector.plot_convergence() # Uncomment to see the plot

    print("\nDemo completed.")
