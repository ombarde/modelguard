"""
Module 1: Weight Drift Analyzer
Compares layer-wise weights between two models.

Detects:
- Which layers changed the most
- Weight distribution shifts
- Bias drift
- Overall model stability
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from modelguard.utils import (
    cosine_similarity,
    l2_distance,
    kl_divergence,
    mean_absolute_change,
    classify_drift,
    get_layer_type,
    format_table,
)


@dataclass
class LayerDriftResult:
    """Drift result for a single layer."""
    layer_name: str
    layer_type: str
    param_type: str           # 'weight' or 'bias'
    l2_diff: float = 0.0
    cosine_sim: float = 1.0
    kl_div: float = 0.0
    mean_abs_change: float = 0.0
    drift_score: float = 0.0  # normalized 0-1
    drift_level: str = ""
    shape: tuple = ()
    num_params: int = 0


@dataclass
class WeightDriftReport:
    """Complete weight drift analysis report."""
    layer_results: List[LayerDriftResult] = field(default_factory=list)
    overall_drift_score: float = 0.0
    overall_drift_level: str = ""
    most_drifted_layers: List[str] = field(default_factory=list)
    stable_layers: List[str] = field(default_factory=list)
    total_params_model_a: int = 0
    total_params_model_b: int = 0
    architecture_match: bool = True
    mismatched_layers: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Print a clean summary of weight drift."""
        lines = []
        lines.append("\n" + "=" * 65)
        lines.append("  📊 WEIGHT DRIFT ANALYSIS REPORT")
        lines.append("=" * 65)

        lines.append(f"\n  Architecture Match: {'✅ Yes' if self.architecture_match else '❌ No'}")
        lines.append(f"  Total Parameters (Model A): {self.total_params_model_a:,}")
        lines.append(f"  Total Parameters (Model B): {self.total_params_model_b:,}")
        lines.append(f"  Overall Drift Score: {self.overall_drift_score:.4f}")
        lines.append(f"  Overall Drift Level: {self.overall_drift_level}")

        if self.mismatched_layers:
            lines.append(f"\n  ⚠️ Mismatched Layers: {', '.join(self.mismatched_layers)}")

        # Layer-by-layer table
        lines.append("\n" + "-" * 65)
        lines.append("  Layer-wise Drift Breakdown:")
        lines.append("-" * 65)

        headers = ["Layer", "Type", "L2 Diff", "Cosine Sim", "Drift", "Status"]
        rows = []

        for result in self.layer_results:
            short_name = result.layer_name
            if len(short_name) > 25:
                short_name = "..." + short_name[-22:]

            rows.append([
                short_name,
                result.param_type[:1].upper(),
                f"{result.l2_diff:.4f}",
                f"{result.cosine_sim:.4f}",
                f"{result.drift_score:.4f}",
                result.drift_level,
            ])

        lines.append(format_table(headers, rows))

        # Top drifted layers
        if self.most_drifted_layers:
            lines.append(f"\n  🔴 Most Drifted: {', '.join(self.most_drifted_layers[:5])}")
        if self.stable_layers:
            lines.append(f"  ✅ Most Stable:  {', '.join(self.stable_layers[:5])}")

        lines.append("=" * 65 + "\n")
        return "\n".join(lines)

    def get_layer_ranking(self) -> List[Tuple[str, float]]:
        """Return layers ranked by drift score (highest first)."""
        ranked = sorted(
            self.layer_results,
            key=lambda x: x.drift_score,
            reverse=True,
        )
        return [(r.layer_name, r.drift_score) for r in ranked]


class WeightDriftAnalyzer:
    """
    Compares weights between two PyTorch models layer by layer.
    
    Metrics computed per layer:
    - L2 norm difference
    - Cosine similarity
    - KL divergence of weight distributions
    - Mean absolute change
    - Normalized drift score (0-1)
    """

    def __init__(self, model_a: torch.nn.Module, model_b: torch.nn.Module):
        self.model_a = model_a
        self.model_b = model_b
        self.params_a = dict(model_a.named_parameters())
        self.params_b = dict(model_b.named_parameters())

    def analyze(self) -> WeightDriftReport:
        """Run complete weight drift analysis."""
        report = WeightDriftReport()

        # Count total parameters
        report.total_params_model_a = sum(
            p.numel() for p in self.model_a.parameters()
        )
        report.total_params_model_b = sum(
            p.numel() for p in self.model_b.parameters()
        )

        # Check architecture match
        report.architecture_match = True
        all_keys_a = set(self.params_a.keys())
        all_keys_b = set(self.params_b.keys())

        if all_keys_a != all_keys_b:
            report.architecture_match = False
            report.mismatched_layers = list(
                all_keys_a.symmetric_difference(all_keys_b)
            )

        # Find common layers to compare
        common_keys = all_keys_a.intersection(all_keys_b)

        # Analyze each layer
        drift_scores = []

        for layer_name in sorted(common_keys):
            param_a = self.params_a[layer_name]
            param_b = self.params_b[layer_name]

            # Check shape match
            if param_a.shape != param_b.shape:
                report.mismatched_layers.append(
                    f"{layer_name} (shape mismatch)"
                )
                continue

            # Determine parameter type
            param_type = "bias" if "bias" in layer_name else "weight"

            # Compute metrics
            l2_diff = l2_distance(param_a, param_b)
            cos_sim = cosine_similarity(param_a, param_b)
            mac = mean_absolute_change(param_a, param_b)

            # KL divergence
            np_a = param_a.detach().cpu().numpy().flatten()
            np_b = param_b.detach().cpu().numpy().flatten()
            kl_div = kl_divergence(np_a, np_b)

            # Compute normalized drift score (0-1)
            drift_score = self._compute_drift_score(
                l2_diff, cos_sim, kl_div, mac, param_a
            )

            layer_result = LayerDriftResult(
                layer_name=layer_name,
                layer_type=get_layer_type(layer_name, self.model_a),
                param_type=param_type,
                l2_diff=l2_diff,
                cosine_sim=cos_sim,
                kl_div=kl_div,
                mean_abs_change=mac,
                drift_score=drift_score,
                drift_level=classify_drift(drift_score),
                shape=tuple(param_a.shape),
                num_params=param_a.numel(),
            )

            report.layer_results.append(layer_result)
            drift_scores.append(drift_score)

        # Overall metrics
        if drift_scores:
            report.overall_drift_score = float(np.mean(drift_scores))
            report.overall_drift_level = classify_drift(
                report.overall_drift_score
            )

            # Find most/least drifted layers
            sorted_results = sorted(
                report.layer_results,
                key=lambda x: x.drift_score,
                reverse=True,
            )

            report.most_drifted_layers = [
                r.layer_name for r in sorted_results[:5]
                if r.drift_score > 0.3
            ]
            report.stable_layers = [
                r.layer_name for r in sorted_results[-5:]
                if r.drift_score < 0.1
            ]

        return report

    def _compute_drift_score(
        self,
        l2_diff: float,
        cos_sim: float,
        kl_div: float,
        mac: float,
        reference_param: torch.Tensor,
    ) -> float:
        """
        Compute normalized drift score (0 to 1).
        
        Combines multiple metrics into one interpretable score.
        
        Formula:
        drift = w1 * (1 - cosine_sim) + w2 * normalized_l2 + w3 * normalized_kl
        """
        # Normalize L2 by the magnitude of the original weights
        ref_norm = torch.norm(
            reference_param.detach().flatten().float()
        ).item()

        if ref_norm > 0:
            normalized_l2 = min(l2_diff / ref_norm, 1.0)
        else:
            normalized_l2 = 0.0

        # Cosine distance (1 - similarity)
        cosine_distance = max(0.0, 1.0 - cos_sim)

        # Normalize KL divergence (cap at 1.0)
        normalized_kl = min(kl_div / 5.0, 1.0)

        # Weighted combination
        drift_score = (
            0.4 * cosine_distance +
            0.35 * normalized_l2 +
            0.25 * normalized_kl
        )

        return min(max(drift_score, 0.0), 1.0)