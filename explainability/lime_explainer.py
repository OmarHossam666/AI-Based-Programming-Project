import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image
import torch
import torchvision.transforms as transforms


class LIMEExplainer:
    """
    Implements Local Interpretable Model-agnostic Explanations (LIME) for images.

    This module fulfills the requirement to provide local interpretability for
    individual MRI scans. By perturbing 'superpixels' (regions) of the image and
    observing how the model's prediction changes, LIME identifies which parts of
    the scan most strongly support the diagnosis.
    """

    def __init__(self, cnn_extractor, feature_mask, ensemble_model):
        """
        Initializes the LIME explainer with the full diagnostic pipeline components.

        Args:
            cnn_extractor: The CNNFeatureExtractor instance to get 7200-dim features.
            feature_mask (np.ndarray): The binary mask of selected features (GA/Heuristic).
            ensemble_model: The trained stacking ensemble model (Stacker).
        """
        self.cnn_extractor = cnn_extractor
        self.feature_mask = feature_mask
        self.ensemble_model = ensemble_model

        # The explainer object itself
        self.explainer = lime_image.LimeImageExplainer(random_state=42)

        # ImageNet normalization parameters (must match the training pipeline)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def _predict_fn(self, images):
        """
        The wrapper function required by LIME. It takes a batch of perturbed
        images and returns their class probabilities.

        This function bridges the gap between raw pixel data and the tabular
        stacking ensemble by running the full feature extraction pipeline.

        Args:
            images (np.ndarray): Batch of images of shape (N, H, W, C).
                                 LIME typically provides these in [0, 255] or [0, 1].

        Returns:
            np.ndarray: Predicted probabilities of shape (N, 2).
        """
        # 1. Preprocessing & Normalization
        # LIME perturbs pixels and might return images in float format.
        # We ensure they are in [0, 1] range first.
        if images.max() > 1.0:
            images = images.astype("float32") / 255.0

        # Transpose from [Batch, H, W, C] to [Batch, C, H, W] for PyTorch
        images_t = np.transpose(images, (0, 3, 1, 2))

        # Convert to tensor and apply ImageNet normalization
        # Note: We apply normalization per-image in the batch
        batch_tensor = torch.from_numpy(images_t).float()
        normalized_batch = torch.stack([self.normalize(img) for img in batch_tensor])

        # 2. CNN Feature Extraction (7200-dim)
        # We pass the normalized batch through the pre-trained CNN
        raw_features = self.cnn_extractor.extract(
            normalized_batch.numpy(), batch_size=len(images)
        )

        # 3. Feature Selection Mask Application
        # We filter the 7200 features down to the optimal subset (e.g., 250 features)
        mask_bool = self.feature_mask == 1
        selected_features = raw_features[:, mask_bool]

        # 4. Stacker Inference
        # The meta-learner predicts the final class probabilities
        return self.ensemble_model.predict_proba(selected_features)

    def explain_instance(self, image_rgb, num_samples=500, top_labels=1):
        """
        Generates a LIME explanation for a single MRI scan.

        Args:
            image_rgb (np.ndarray): The input MRI image (H, W, C).
            num_samples (int): Number of perturbations to run. Higher is more accurate but slower.
            top_labels (int): Number of top labels to explain.

        Returns:
            ImageExplanation: The LIME explanation object.
        """
        print(f"Running LIME with {num_samples} samples...")
        return self.explainer.explain_instance(
            image_rgb,
            self._predict_fn,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
        )

    def save_explanation(self, explanation, save_path, label_idx=None):
        """
        Utility to visualize and save the LIME overlay.

        Args:
            explanation: The object returned by explain_instance.
            save_path (str): File path to save the resulting plot.
            label_idx (int): The class label to explain. If None, uses the top predicted label.
        """
        if label_idx is None:
            label_idx = explanation.top_labels[0]

        # Get the image and mask for the specified label
        # positive_only=False allows us to see regions that 'negatively' impact the class
        temp, mask = explanation.get_image_and_mask(
            label_idx, positive_only=False, num_features=5, hide_rest=False
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(mark_boundaries(temp / temp.max(), mask))
        plt.title(f"LIME Explainer Overlay - Class {label_idx}")
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"LIME explanation saved to {save_path}")
