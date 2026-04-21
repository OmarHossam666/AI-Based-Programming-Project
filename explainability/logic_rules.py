import numpy as np
from dataclasses import dataclass
from typing import List, Callable, Dict, Any


@dataclass
class Rule:
    """Represents a single symbolic logic rule."""

    name: str
    condition: Callable[[np.ndarray], bool]
    explanation: str


@dataclass
class RuleResult:
    """Holds the results after applying the rule set to a sample."""

    triggered_rules: List[str]
    risk_level: str
    explanation_text: str


class RuleBasedAssistant:
    """
    Integrates ML predictions with symbolic reasoning.

    This assistant builds a set of IF-THEN rules based on the most globally
    important CNN features identified by SHAP. It acts as an explainability
    layer, providing human-readable justifications for clinical predictions.
    """

    def __init__(self):
        self.thresholds: Dict[int, Dict[str, float]] = {}
        self.RULE_SET: List[Rule] = []

    def extract_thresholds(
        self, X: np.ndarray, y: np.ndarray, shap_top_features: List[int]
    ):
        """
        Derives per-feature thresholds using the training data.

        The threshold is calculated as the midpoint between the mean activation
        of the Healthy class and the Tumor class for each top feature.

        Args:
            X (np.ndarray): Training feature matrix.
            y (np.ndarray): Training labels (0 = Healthy, 1 = Tumor).
            shap_top_features (List[int]): Column indices of the top SHAP features.
        """
        print("Extracting symbolic thresholds from training data...")
        self.thresholds = {}

        for feat_idx in shap_top_features:
            healthy_vals = X[y == 0, feat_idx]
            tumor_vals = X[y == 1, feat_idx]

            mean_healthy = np.mean(healthy_vals)
            mean_tumor = np.mean(tumor_vals)

            # Midpoint separating the two classes
            threshold = (mean_healthy + mean_tumor) / 2.0

            # Record whether a tumor is indicated by values above or below the threshold
            tumor_is_higher = mean_tumor > mean_healthy

            self.thresholds[feat_idx] = {
                "value": threshold,
                "tumor_is_higher": tumor_is_higher,
            }

        # Build the dynamic rule set based on the extracted thresholds
        self._build_rules(shap_top_features)

    def _check_feature(self, x_flat: np.ndarray, feat_idx: int) -> bool:
        """Helper to evaluate if a single feature triggers its tumor-associated threshold."""
        if feat_idx not in self.thresholds:
            return False

        t_info = self.thresholds[feat_idx]
        val = x_flat[feat_idx]

        if t_info["tumor_is_higher"]:
            return val > t_info["value"]
        else:
            return val < t_info["value"]

    def _build_rules(self, top_features: List[int]):
        """
        Derives 5-8 IF-THEN rules based on the top SHAP features and domain assumptions.
        """
        self.RULE_SET = []

        # We need at least 5 features to build our standard rule set
        if len(top_features) < 5:
            print(
                "Warning: Not enough top features provided to build the full rule set."
            )
            return

        f0, f1, f2, f3, f4 = top_features[:5]
        f_extra = top_features[5:8] if len(top_features) >= 8 else []

        # Rule 1: Primary Core Pattern (Requires both top 2 features)
        self.RULE_SET.append(
            Rule(
                name="R1",
                condition=lambda x: self._check_feature(x, f0)
                and self._check_feature(x, f1),
                explanation=f"elevated activity in primary CNN features {f0} and {f1} (Core Tumor Pattern)",
            )
        )

        # Rule 2: Secondary Structural Pattern
        self.RULE_SET.append(
            Rule(
                name="R2",
                condition=lambda x: self._check_feature(x, f2),
                explanation=f"anomalous spatial activation in CNN feature {f2} (Secondary Structural Irregularity)",
            )
        )

        # Rule 3: Asymmetric Border Pattern
        self.RULE_SET.append(
            Rule(
                name="R3",
                condition=lambda x: self._check_feature(x, f3)
                and self._check_feature(x, f0),
                explanation=f"co-activation of CNN features {f3} and {f0} (Asymmetric Border Profile)",
            )
        )

        # Rule 4: High-Contrast Texture Anomaly
        self.RULE_SET.append(
            Rule(
                name="R4",
                condition=lambda x: self._check_feature(x, f4),
                explanation=f"irregular texture gradient detected in CNN feature {f4}",
            )
        )

        # Rule 5: Critical Multi-Feature Co-occurrence
        self.RULE_SET.append(
            Rule(
                name="R5",
                condition=lambda x: sum(
                    [self._check_feature(x, f) for f in [f0, f1, f2, f3, f4]]
                )
                >= 4,
                explanation=f"widespread anomalous activation across 4 or more primary CNN spatial features",
            )
        )

        # Optional Extended Rules (if we have more than 5 features)
        if len(f_extra) >= 1:
            self.RULE_SET.append(
                Rule(
                    name="R6",
                    condition=lambda x: self._check_feature(x, f_extra[0]),
                    explanation=f"peripheral mass indicator present in CNN feature {f_extra[0]}",
                )
            )

    def apply_rules(self, x_sample: np.ndarray) -> RuleResult:
        """
        Applies the symbolic rule set to a single MRI feature vector.

        Args:
            x_sample (np.ndarray): The feature vector for a single scan.

        Returns:
            RuleResult: The triggered rules, estimated risk level, and explanation text.
        """
        x_flat = x_sample.flatten()
        triggered_names = []
        explanations = []

        for rule in self.RULE_SET:
            if rule.condition(x_flat):
                triggered_names.append(rule.name)
                explanations.append(rule.explanation)

        # Determine symbolic risk level
        if len(triggered_names) >= 3:
            risk_level = "High Risk"
        elif len(triggered_names) >= 1:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Low Risk"

        exp_text = (
            " and ".join(explanations)
            if explanations
            else "No prominent tumor-associated patterns detected"
        )

        return RuleResult(
            triggered_rules=triggered_names,
            risk_level=risk_level,
            explanation_text=exp_text,
        )

    def generate_explanation(
        self, ml_prediction: int, ml_confidence: float, rule_result: RuleResult
    ) -> str:
        """
        Merges the ML confidence score with symbolic reasoning into a structured,
        human-readable clinical explanation string.
        """
        pred_label = "Tumor" if ml_prediction == 1 else "Healthy"

        if not rule_result.triggered_rules:
            return (
                f"Model predicts {pred_label} (confidence {ml_confidence:.0%}). "
                f"However, no symbolic rules triggered: {rule_result.explanation_text}. "
                f"Manual radiologist review is recommended to verify this prediction."
            )

        rules_str = (
            " and ".join(rule_result.triggered_rules)
            if len(rule_result.triggered_rules) < 3
            else ", ".join(rule_result.triggered_rules)
        )

        return (
            f"Model predicts {pred_label} (confidence {ml_confidence:.0%}). "
            f"Rule {rules_str} triggered: {rule_result.explanation_text} "
            f"— consistent with tumour-associated spatial patterns."
        )
