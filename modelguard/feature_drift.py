"""
Module 4: Feature Influence Drift Analyzer
Compares which input features each model relies on.

Uses gradient-based attribution (no external dependency like SHAP needed).

Detects:
- Feature importance shift between models
- Which features gained/lost influence
- Feature reliance concentration changes

This is critical for understanding WHY predictions changed.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from modelguard.utils import (
    validate_model,
    validate_dataset,
    classify_drift,
    format_table,
)


@dataclass
class FeatureResult:
    """Drift result for a single feature."""
    feature_id: int
    feature_name: str
    importance_model_a: float = 0.0
    importance_model_b: float = 0.0
    importance_change: float = 0.0       # absolute change
    relative_change: float = 0.0         # percentage change
    rank_model_a: int = 0
    rank_model_b: int = 0
    rank_change: int = 0


@dataclass
class FeatureDriftReport:
    """Complete feature influence drift report."""
    feature_results: List[FeatureResult] = field(default_factory=list)
    total_features: int = 0
    overall_feature_drift: float = 0.0
    overall_drift_level: str = ""

    # Top movers
    top_gained: List[str] = field(default_factory=list)    # features that gained importance
    top_lost: List[str] = field(default_factory=list)      # features that lost importance
    top_rank_changes: List[str] = field(default_factory=list)

    # Concentration metrics
    concentration_model_a: float = 0.0   # how concentrated importance is (Gini-like)
    concentration_model_b: float = 0.0
    concentration_change: float = 0.0

    # Raw importance arrays
    importances_model_a: Optional[np.ndarray] = None
    importances_model_b: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Print clean summary of feature drift."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  🎯 FEATURE INFLUENCE DRIFT REPORT")
        lines.append("=" * 70)

        lines.append(f"\n  Total Features Analyzed:  {self.total_features}")
        lines.append(f"  Overall Feature Drift:    {self.overall_feature_drift:.4f}")
        lines.append(f"  Feature Drift Level:      {self.overall_drift_level}")

        lines.append(f"\n  Importance Concentration (Model A): {self.concentration_model_a:.4f}")
        lines.append(f"  Importance Concentration (Model B): {self.concentration_model_b:.4f}")
        lines.append(f"  Concentration Change:               {self.concentration_change:+.4f}")

        # Feature table
        lines.append("\n" + "-" * 70)
        lines.append("  Feature-wise Importance Comparison:")
        lines.append("-" * 70)

        headers = [
            "Feature", "Imp A", "Imp B",
            "Change", "Rank A", "Rank B", "Status"
        ]
        rows = []

        # Sort by absolute change
        sorted_features = sorted(
            self.feature_results,
            key=lambda x: abs(x.importance_change),
            reverse=True,
        )

        for fr in sorted_features:
            if fr.importance_change > 0.01:
                status = "📈 GAINED"
            elif fr.importance_change < -0.01:
                status = "📉 LOST"
            else:
                status = "➡️ STABLE"

            rows.append([
                fr.feature_name,
                f"{fr.importance_model_a:.4f}",
                f"{fr.importance_model_b:.4f}",
                f"{fr.importance_change:+.4f}",
                str(fr.rank_model_a),
                str(fr.rank_model_b),
                status,
            ])

        lines.append(format_table(headers, rows))

        # Top movers
        if self.top_gained:
            lines.append(f"\n  📈 Top Gained Importance: {', '.join(self.top_gained[:5])}")
        if self.top_lost:
            lines.append(f"  📉 Top Lost Importance:   {', '.join(self.top_lost[:5])}")
        if self.top_rank_changes:
            lines.append(f"  🔄 Biggest Rank Changes:  {', '.join(self.top_rank_changes[:5])}")

        lines.append("=" * 70 + "\n")
        return "\n".join(lines)

    def get_feature_ranking_comparison(self) -> Dict[str, Dict]:
        """Get detailed ranking comparison for each feature."""
        comparison = {}
        for fr in self.feature_results:
            comparison[fr.feature_name] = {
                "importance_a": fr.importance_model_a,
                "importance_b": fr.importance_model_b,
                "change": fr.importance_change,
                "rank_a": fr.rank_model_a,
                "rank_b": fr.rank_model_b,
                "rank_change": fr.rank_change,
            }
        return comparison


class GradientAttributor:
    """
    Compute feature importance using gradient-based attribution.
    
    Method: Input x Gradient
    - Compute gradient of output with respect to input
    - Multiply by input values
    - Average absolute values across samples
    
    This is lightweight, requires no external libraries,
    and works with any differentiable PyTorch model.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model

    def compute_importance(
        self,
        dataset: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Compute feature importance scores.

        Args:
            dataset: Input tensor (N, features)
            target_class: Which output class to explain (None = predicted class)
            batch_size: Batch size for computation

        Returns:
            Array of importance scores per feature
        """
        self.model.eval()
        all_attributions = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size].clone().detach()

            device = next(self.model.parameters()).device
            batch = batch.to(device)
            batch.requires_grad_(True)

            # Forward pass
            output = self.model(batch)

            # Select target
            if target_class is not None:
                target_output = output[:, target_class]
            else:
                # Use predicted class for each sample
                predicted = output.argmax(dim=1)
                target_output = output.gather(
                    1, predicted.unsqueeze(1)
                ).squeeze(1)

            # Backward pass
            self.model.zero_grad()
            target_output.sum().backward()

            # Input x Gradient attribution
            if batch.grad is not None:
                attribution = (batch * batch.grad).abs().detach().cpu().numpy()
                all_attributions.append(attribution)

            # Clear gradients
            batch.requires_grad_(False)

        if not all_attributions:
            # Fallback: return uniform importance
            n_features = dataset.shape[1] if dataset.dim() > 1 else dataset.shape[0]
            return np.ones(n_features) / n_features

        # Average across all samples
        all_attributions = np.concatenate(all_attributions, axis=0)
        importance = np.mean(all_attributions, axis=0)

        # Handle multi-dimensional features (flatten)
        if importance.ndim > 1:
            importance = importance.reshape(-1)

        # Normalize to sum to 1
        total = importance.sum()
        if total > 0:
            importance = importance / total

        return importance


class FeatureDriftAnalyzer:
    """
    Compares feature importance between two models.

    Process:
    1. Compute gradient-based feature importance for both models
    2. Compare importance distributions
    3. Identify features that gained/lost influence
    4. Measure overall feature reliance drift

    This answers: "Is the model relying on different features now?"
    """

    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        feature_names: Optional[List[str]] = None,
    ):
        validate_model(model_a)
        validate_model(model_b)

        self.model_a = model_a
        self.model_b = model_b
        self.feature_names = feature_names

    def analyze(
        self,
        dataset: torch.Tensor,
        target_class: Optional[int] = None,
        batch_size: int = 64,
    ) -> FeatureDriftReport:
        """
        Run feature drift analysis.

        Args:
            dataset: Input tensor (N, features)
            target_class: Which class to explain (None = predicted)
            batch_size: Batch size for gradient computation

        Returns:
            FeatureDriftReport
        """
        dataset = validate_dataset(dataset)

        # Compute feature importance for both models
        attributor_a = GradientAttributor(self.model_a)
        attributor_b = GradientAttributor(self.model_b)

        importance_a = attributor_a.compute_importance(
            dataset, target_class, batch_size
        )
        importance_b = attributor_b.compute_importance(
            dataset, target_class, batch_size
        )

        # Ensure same length
        min_len = min(len(importance_a), len(importance_b))
        importance_a = importance_a[:min_len]
        importance_b = importance_b[:min_len]

        # Build report
        report = self._build_report(importance_a, importance_b)

        return report

    def _build_report(
        self,
        importance_a: np.ndarray,
        importance_b: np.ndarray,
    ) -> FeatureDriftReport:
        """Build the feature drift report."""
        report = FeatureDriftReport()
        n_features = len(importance_a)
        report.total_features = n_features

        # Store raw importances
        report.importances_model_a = importance_a
        report.importances_model_b = importance_b

        # Generate feature names if not provided
        feature_names = self.feature_names or [
            f"Feature_{i}" for i in range(n_features)
        ]

        # Get rankings (1 = most important)
        ranks_a = self._get_ranks(importance_a)
        ranks_b = self._get_ranks(importance_b)

        # Build per-feature results
        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"

            fr = FeatureResult(
                feature_id=i,
                feature_name=name,
                importance_model_a=float(importance_a[i]),
                importance_model_b=float(importance_b[i]),
                importance_change=float(importance_b[i] - importance_a[i]),
                rank_model_a=int(ranks_a[i]),
                rank_model_b=int(ranks_b[i]),
                rank_change=int(ranks_a[i] - ranks_b[i]),  # positive = improved rank
            )

            # Relative change
            if importance_a[i] > 1e-8:
                fr.relative_change = float(
                    (importance_b[i] - importance_a[i]) / importance_a[i]
                )
            else:
                fr.relative_change = 0.0

            report.feature_results.append(fr)

        # --- Overall feature drift ---
        report.overall_feature_drift = self._compute_feature_drift(
            importance_a, importance_b
        )
        report.overall_drift_level = classify_drift(report.overall_feature_drift)

        # --- Concentration (Gini-like) ---
        report.concentration_model_a = self._compute_concentration(importance_a)
        report.concentration_model_b = self._compute_concentration(importance_b)
        report.concentration_change = (
            report.concentration_model_b - report.concentration_model_a
        )

        # --- Top movers ---
        sorted_by_gain = sorted(
            report.feature_results,
            key=lambda x: x.importance_change,
            reverse=True,
        )

        report.top_gained = [
            fr.feature_name for fr in sorted_by_gain[:5]
            if fr.importance_change > 0.01
        ]

        report.top_lost = [
            fr.feature_name for fr in sorted_by_gain[-5:]
            if fr.importance_change < -0.01
        ]

        sorted_by_rank = sorted(
            report.feature_results,
            key=lambda x: abs(x.rank_change),
            reverse=True,
        )

        report.top_rank_changes = [
            f"{fr.feature_name} ({fr.rank_change:+d})"
            for fr in sorted_by_rank[:5]
            if abs(fr.rank_change) > 0
        ]

        return report

    def _get_ranks(self, importance: np.ndarray) -> np.ndarray:
        """Convert importance scores to ranks (1 = most important)."""
        # argsort gives indices that would sort ascending
        # we want descending, so negate
        order = np.argsort(-importance)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(importance) + 1)
        return ranks

    def _compute_feature_drift(
        self,
        importance_a: np.ndarray,
        importance_b: np.ndarray,
    ) -> float:
        """
        Compute overall feature drift score (0 to 1).

        Uses:
        1. L1 distance between importance distributions
        2. Rank correlation change
        3. Top-k overlap
        """
        # L1 distance (since both sum to 1, max distance = 2)
        l1_distance = np.sum(np.abs(importance_a - importance_b))
        normalized_l1 = min(l1_distance / 2.0, 1.0)

        # Rank correlation (Spearman-like)
        ranks_a = self._get_ranks(importance_a)
        ranks_b = self._get_ranks(importance_b)
        n = len(ranks_a)

        if n > 1:
            rank_diff_sq = np.sum((ranks_a - ranks_b) ** 2)
            max_rank_diff = (n * (n ** 2 - 1)) / 3.0
            rank_correlation = 1.0 - (rank_diff_sq / max(max_rank_diff, 1.0))
            rank_distance = max(0.0, 1.0 - rank_correlation) / 2.0
        else:
            rank_distance = 0.0

        # Top-k overlap (top 20% features)
        k = max(1, n // 5)
        top_k_a = set(np.argsort(-importance_a)[:k])
        top_k_b = set(np.argsort(-importance_b)[:k])
        overlap = len(top_k_a & top_k_b) / k
        top_k_distance = 1.0 - overlap

        # Weighted combination
        score = (
            0.40 * normalized_l1 +
            0.30 * rank_distance +
            0.30 * top_k_distance
        )

        return min(max(score, 0.0), 1.0)

    def _compute_concentration(self, importance: np.ndarray) -> float:
        """
        Compute importance concentration (Gini coefficient).
        
        0 = perfectly equal importance across all features
        1 = all importance on one feature
        """
        sorted_imp = np.sort(importance)
        n = len(sorted_imp)

        if n == 0 or sorted_imp.sum() == 0:
            return 0.0

        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_imp)) / (n * np.sum(sorted_imp)) - (n + 1) / n

        return float(max(0.0, min(gini, 1.0)))