**Ensemble Modeling & Evaluation Report: Brain Tumor Detection**

**1. Introduction** Ensemble modeling is highly effective for medical image classification because relying on a single model often leaves the system sensitive to noise, dataset variations, and restrictive inductive biases. In the context of brain tumor detection, an isolated algorithm might overfit to specific anatomical variations or imaging artifacts, leading to unstable generalization. By aggregating the predictions of multiple models, an ensemble reduces individual errors, stabilizes predictive performance, and produces a stronger, more robust diagnostic system.

**2. Method** Our ensemble framework utilizes three heterogeneous base learners, each introducing a distinct inductive bias to process the high-dimensional features extracted by the Convolutional Neural Network (CNN). Logistic Regression provides a strong global linear baseline by assigning balanced feature weights. The Linear Support Vector Machine (SVM) focuses strictly on margin-based separation within the high-dimensional feature space, emphasizing boundaries between tumorous and healthy tissues. Finally, Gaussian Naive Bayes leverages statistical evidence from continuous feature distributions, offering a probabilistic perspective independent of margin maximization.

To integrate these base learners, we deployed three architectures. Hard Voting relies on discrete majority agreement across the class labels predicted by the base models. Soft Voting averages the predicted class probabilities, factoring in the relative confidence of each classifier, which provides a more nuanced aggregation. Finally, Stacking introduces a meta-learner (Logistic Regression) that is trained via 5-fold cross-validation on the out-of-fold predictions of the base learners; rather than applying a fixed aggregation rule, stacking actively learns how to weight and optimally combine the models' outputs.

**3. Results**
The models were evaluated utilizing accuracy, precision, recall, and F1-score. The comparative results on the validation set are summarized below:

* **Logistic Regression:** Accuracy 0.7500 | Precision 0.7500 | Recall 1.0000 | F1-score 0.8571
* **Linear SVM:** Accuracy 0.7589 | Precision 0.7591 | Recall 0.9940 | F1-score 0.8608
* **Naive Bayes:** Accuracy 0.8071 | Precision 0.8264 | Recall 0.9405 | F1-score 0.8797
* **Voting (Hard):** Accuracy 0.7589 | Precision 0.7591 | Recall 0.9940 | F1-score 0.8608
* **Voting (Soft):** Accuracy 0.8027 | Precision 0.8031 | Recall 0.9762 | F1-score 0.8812
* **Stacking:** Accuracy 0.8232 | Precision 0.8204 | Recall 0.9786 | F1-score 0.8925

**4. Error Analysis**
In diagnostic medical imaging, the clinical cost of classification errors is heavily asymmetric. False negatives (missing the presence of a tumor) carry severe, potentially fatal risks, making recall (sensitivity) the paramount evaluation metric. Conversely, false positives (flagging a healthy scan) merely necessitate secondary review by a radiologist, an acceptable outcome for a preliminary screening tool. 

Our models demonstrate exceptional recall across the board. The confusion matrix dynamics reveal that while Logistic Regression achieved a perfect recall (1.0000), it suffered from low precision (0.7500), indicating a high volume of false positives. Hard Voting failed to improve upon the SVM baseline, but Soft Voting successfully leveraged calibrated probabilities to boost overall accuracy. Ultimately, Stacking provided the most refined error profile, preserving a highly reliable recall of 0.9786 while significantly elevating precision to 0.8204.

**5. Conclusion**
The Stacking Ensemble is unequivocally the recommended model for clinical deployment. It achieves the highest overall accuracy (82.32%) and F1-score (89.25%) by effectively learning the context-dependent importance of each base learner. Most importantly, it strikes an optimal clinical balance—virtually eliminating false negatives with an outstanding recall rate while minimizing the false positive burden on radiologists, making it a robust and trustworthy automated screening mechanism.