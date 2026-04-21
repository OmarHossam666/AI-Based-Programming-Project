**Technical Note: Data Engineering and Feature Selection Module**

**1. Feature Selection Strategy**

In the development of the Brain Tumor Detection system, relying on raw MRI pixels for classification is suboptimal due to high dimensionality, spatial variance, and the presence of irrelevant background noise. Instead, we utilize a custom Convolutional Neural Network (CNN) as a feature extractor. The CNN’s convolutional and pooling layers automatically detect hierarchical spatial patterns—such as edges, textures, and anatomical shapes—resulting in a flattened $32 \times 15 \times 15$ tensor, yielding a 7200-dimensional feature vector per image.

However, utilizing all 7200 CNN-extracted features directly introduces the curse of dimensionality, increases computational overhead, and risks overfitting due to redundant or uninformative feature maps. To address this, we apply algorithmic feature selection on top of the extracted representations to isolate the most discriminative subset. We implemented two distinct strategies for comparison:

* **Greedy Heuristic Selection:** This deterministic approach iteratively evaluates and adds the single feature that most improves classification accuracy. While computationally efficient and fast, its greedy nature means it cannot re-evaluate past additions, leaving it highly susceptible to converging on a local optimum.
* **Genetic Algorithm (GA):** The GA is a stochastic, population-based metaheuristic inspired by biological evolution. By utilizing mechanisms such as crossover (combining subsets) and mutation (randomly flipping feature inclusions), the GA explores a vastly broader feature space. This allows it to escape local optima and often discover globally superior feature combinations, albeit at the cost of significantly higher computational time.

**2. Performance vs. Baseline**

To quantify the efficacy of our feature selection pipeline, we established a baseline performance by evaluating the full 7200-dimensional dataset using a lightweight classifier. Operating on the complete feature space typically yields strong baseline accuracy but suffers from excessive memory consumption and slower inference. 

Following the application of our selection algorithms, both the heuristic and genetic methods successfully reduced the feature space to an optimal subset of approximately 250 features. Despite discarding over 96% of the original feature matrix, the reduced subsets demonstrated performance that matched or slightly exceeded the baseline accuracy. For instance, the GA-selected features often achieve superior stabilization across validation folds due to the algorithm's thorough exploration of synergistic feature interactions.

This drastic reduction in dimensionality has profound implications for the downstream Ensemble Modeling module. By feeding our voting and stacking ensembles a condensed, highly informative, and noise-free feature matrix, we significantly reduce model complexity. This ensures faster training times , mitigates the risk of individual base learners memorizing noise (overfitting), and ultimately produces a more robust, generalizable, and clinically viable diagnostic tool.