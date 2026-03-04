"""
Module 3: Activation Drift Analyzer
Captures and compares intermediate layer activations between two models.

This is the KEY DIFFERENTIATOR of ModelGuard.
Most tools compare inputs/outputs only.
We compare INTERNAL REPRESENTATIONS.

Detects:
- Which internal layers changed behavior
- Activation distribution shifts
- Dead neuron changes
- Representation divergence
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict

from modelguard.utils import (
    cosine_similarity,
    l2_distance,
    kl_divergence,
    classify_drift,
    format_table,
    validate_model,
    validate_dataset,
)


@dataclass
class LayerActivationResult:
    """Activation drift result for a single layer."""
    layer_name: str
    layer_type: str
    output_shape: tuple = ()

    # Distribution metrics
    mean_activation_a: float = 0.0
    mean_activation_b: float = 0.0
    std_activation_a: float = 0.0
    std_activation_b: float = 0.0

    # Comparison metrics
    mean_cosine_similarity: float = 1.0
    mean_l2_distance: float = 0.0
    kl_divergence: float = 0.0
    activation_drift_score: float = 0.0
    drift_level: str = ""

    # Neuron-level analysis
    total_neurons: int = 0
    dead_neurons_a: int = 0     # neurons always outputting 0
    dead_neurons_b: int = 0
    dead_neuron_change: int = 0

    # Entropy
    entropy_a: float = 0.0
    entropy_b: float = 0.0
    entropy_change: float = 0.0


@dataclass
class ActivationDriftReport:
    """Complete activation drift analysis report."""
    layer_results: List[LayerActivationResult] = field(default_factory=list)
    overall_activation_drift: float = 0.0
    overall_drift_level: str = ""
    most_drifted_layers: List[str] = field(default_factory=list)
    stable_layers: List[str] = field(default_factory=list)
    total_layers_analyzed: int = 0
    total_samples: int = 0

    def summary(self) -> str:
        """Print clean summary of activation drift."""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  🧠 ACTIVATION DRIFT ANALYSIS REPORT")
        lines.append("=" * 70)

        lines.append(f"\n  Layers Analyzed:          {self.total_layers_analyzed}")
        lines.append(f"  Samples Used:             {self.total_samples}")
        lines.append(f"  Overall Activation Drift: {self.overall_activation_drift:.4f}")
        lines.append(f"  Overall Drift Level:      {self.overall_drift_level}")

        # Layer-by-layer table
        lines.append("\n" + "-" * 70)
        lines.append("  Layer-wise Activation Drift:")
        lines.append("-" * 70)

        headers = [
            "Layer", "Type", "Cosine Sim",
            "L2 Dist", "KL Div", "Drift", "Status"
        ]
        rows = []

        for result in self.layer_results:
            short_name = result.layer_name
            if len(short_name) > 20:
                short_name = "..." + short_name[-17:]

            rows.append([
                short_name,
                result.layer_type[:8],
                f"{result.mean_cosine_similarity:.4f}",
                f"{result.mean_l2_distance:.4f}",
                f"{result.kl_divergence:.4f}",
                f"{result.activation_drift_score:.4f}",
                result.drift_level,
            ])

        lines.append(format_table(headers, rows))

        # Dead neuron analysis
        has_dead = any(
            r.dead_neurons_a > 0 or r.dead_neurons_b > 0
            for r in self.layer_results
        )
        if has_dead:
            lines.append("\n" + "-" * 70)
            lines.append("  Dead Neuron Analysis:")
            lines.append("-" * 70)
            for result in self.layer_results:
                if result.dead_neurons_a > 0 or result.dead_neurons_b > 0:
                    change_str = ""
                    if result.dead_neuron_change > 0:
                        change_str = f"  (+{result.dead_neuron_change} new dead)"
                    elif result.dead_neuron_change < 0:
                        change_str = f"  ({result.dead_neuron_change} recovered)"

                    lines.append(
                        f"  {result.layer_name}: "
                        f"Model A={result.dead_neurons_a}, "
                        f"Model B={result.dead_neurons_b}"
                        f"{change_str}"
                    )

        # Most drifted
        if self.most_drifted_layers:
            lines.append(f"\n  🔴 Most Drifted Layers: {', '.join(self.most_drifted_layers[:5])}")
        if self.stable_layers:
            lines.append(f"  ✅ Most Stable Layers:  {', '.join(self.stable_layers[:5])}")

        lines.append("=" * 70 + "\n")
        return "\n".join(lines)

    def get_layer_ranking(self) -> List[Tuple[str, float]]:
        """Return layers ranked by activation drift (highest first)."""
        ranked = sorted(
            self.layer_results,
            key=lambda x: x.activation_drift_score,
            reverse=True,
        )
        return [(r.layer_name, r.activation_drift_score) for r in ranked]

    def get_blame(self) -> str:
        """
        Generate a 'git blame' style analysis.
        Identifies which layer is most responsible for output changes.
        """
        if not self.layer_results:
            return "No layers analyzed."

        ranked = sorted(
            self.layer_results,
            key=lambda x: x.activation_drift_score,
            reverse=True,
        )

        top = ranked[0]

        blame_lines = []
        blame_lines.append("\n  🔍 BLAME ANALYSIS")
        blame_lines.append("  " + "-" * 50)
        blame_lines.append(
            f"  Primary drift source: {top.layer_name} "
            f"({top.layer_type})"
        )
        blame_lines.append(
            f"  Drift score: {top.activation_drift_score:.4f} "
            f"({top.drift_level})"
        )
        blame_lines.append(
            f"  Activation mean shifted: "
            f"{top.mean_activation_a:.4f} → {top.mean_activation_b:.4f}"
        )
        blame_lines.append(
            f"  Activation std shifted:  "
            f"{top.std_activation_a:.4f} → {top.std_activation_b:.4f}"
        )

        if top.dead_neuron_change != 0:
            blame_lines.append(
                f"  Dead neuron change: {top.dead_neuron_change:+d}"
            )

        if len(ranked) > 1:
            blame_lines.append(f"\n  Secondary contributors:")
            for r in ranked[1:3]:
                if r.activation_drift_score > 0.1:
                    blame_lines.append(
                        f"    - {r.layer_name}: "
                        f"drift={r.activation_drift_score:.4f}"
                    )

        return "\n".join(blame_lines)


class ActivationCapturer:
    """
    Hooks into PyTorch model to capture intermediate activations.
    Uses forward hooks — non-invasive, does not modify the model.
    """

    def __init__(self, model: torch.nn.Module, layer_names: Optional[List[str]] = None):
        self.model = model
        self.activations = OrderedDict()
        self.hooks = []
        self.layer_names = layer_names
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""
        for name, module in self.model.named_modules():
            # Skip the top-level module itself
            if name == "":
                continue

            # Skip container modules (they don't produce activations)
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                continue

            # If specific layers requested, filter
            if self.layer_names and name not in self.layer_names:
                continue

            hook = module.register_forward_hook(
                self._create_hook(name)
            )
            self.hooks.append(hook)

    def _create_hook(self, name: str):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                self.activations[name] = output[0].detach().cpu()
        return hook_fn

    def capture(self, dataset: torch.Tensor, batch_size: int = 64) -> Dict[str, np.ndarray]:
        """
        Run dataset through model and capture all activations.

        Returns dict: layer_name -> activations array
        """
        self.model.eval()
        all_activations = {}

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]

                device = next(self.model.parameters()).device
                batch = batch.to(device)

                # Clear previous activations
                self.activations.clear()

                # Forward pass triggers hooks
                _ = self.model(batch)

                # Collect activations
                for layer_name, activation in self.activations.items():
                    act_np = activation.numpy()

                    if layer_name not in all_activations:
                        all_activations[layer_name] = []
                    all_activations[layer_name].append(act_np)

        # Concatenate batches
        result = {}
        for layer_name, act_list in all_activations.items():
            result[layer_name] = np.concatenate(act_list, axis=0)

        return result

    def remove_hooks(self):
        """Remove all hooks from model."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class ActivationDriftAnalyzer:
    """
    Compares internal activations between two models.

    Process:
    1. Hook into both models
    2. Run same dataset through both
    3. Compare activations at each layer
    4. Report which internal representations changed

    This is what makes ModelGuard unique.
    """

    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        layer_names: Optional[List[str]] = None,
    ):
        validate_model(model_a)
        validate_model(model_b)

        self.model_a = model_a
        self.model_b = model_b
        self.layer_names = layer_names

    def analyze(
        self,
        dataset: torch.Tensor,
        batch_size: int = 64,
    ) -> ActivationDriftReport:
        """
        Run activation drift analysis.

        Args:
            dataset: Input tensor (N, ...)
            batch_size: Batch size for inference

        Returns:
            ActivationDriftReport
        """
        dataset = validate_dataset(dataset)

        # Capture activations from both models
        capturer_a = ActivationCapturer(self.model_a, self.layer_names)
        capturer_b = ActivationCapturer(self.model_b, self.layer_names)

        try:
            activations_a = capturer_a.capture(dataset, batch_size)
            activations_b = capturer_b.capture(dataset, batch_size)
        finally:
            capturer_a.remove_hooks()
            capturer_b.remove_hooks()

        # Compare activations
        report = self._compare_activations(
            activations_a, activations_b, len(dataset)
        )

        return report

    def _compare_activations(
        self,
        activations_a: Dict[str, np.ndarray],
        activations_b: Dict[str, np.ndarray],
        n_samples: int,
    ) -> ActivationDriftReport:
        """Compare activations layer by layer."""
        report = ActivationDriftReport()
        report.total_samples = n_samples

        # Find common layers
        common_layers = set(activations_a.keys()) & set(activations_b.keys())
        report.total_layers_analyzed = len(common_layers)

        drift_scores = []

        for layer_name in sorted(common_layers):
            act_a = activations_a[layer_name]
            act_b = activations_b[layer_name]

            # Skip if shapes don't match
            if act_a.shape != act_b.shape:
                continue

            result = self._analyze_layer(layer_name, act_a, act_b)
            report.layer_results.append(result)
            drift_scores.append(result.activation_drift_score)

        # Overall metrics
        if drift_scores:
            report.overall_activation_drift = float(np.mean(drift_scores))
            report.overall_drift_level = classify_drift(
                report.overall_activation_drift
            )

            sorted_results = sorted(
                report.layer_results,
                key=lambda x: x.activation_drift_score,
                reverse=True,
            )

            report.most_drifted_layers = [
                r.layer_name for r in sorted_results[:5]
                if r.activation_drift_score > 0.3
            ]
            report.stable_layers = [
                r.layer_name for r in sorted_results[-5:]
                if r.activation_drift_score < 0.1
            ]

        return report

    def _analyze_layer(
        self,
        layer_name: str,
        act_a: np.ndarray,
        act_b: np.ndarray,
    ) -> LayerActivationResult:
        """Analyze activation drift for a single layer."""
        result = LayerActivationResult(
            layer_name=layer_name,
            layer_type=self._infer_layer_type(layer_name),
            output_shape=act_a.shape[1:],  # exclude batch dim
        )

        # Flatten activations per sample for comparison
        flat_a = act_a.reshape(act_a.shape[0], -1)
        flat_b = act_b.reshape(act_b.shape[0], -1)

        # --- Distribution Statistics ---
        result.mean_activation_a = float(np.mean(act_a))
        result.mean_activation_b = float(np.mean(act_b))
        result.std_activation_a = float(np.std(act_a))
        result.std_activation_b = float(np.std(act_b))

        # --- Cosine Similarity (per sample, then average) ---
        cosine_sims = []
        for i in range(len(flat_a)):
            vec_a = torch.tensor(flat_a[i], dtype=torch.float32)
            vec_b = torch.tensor(flat_b[i], dtype=torch.float32)
            cos = cosine_similarity(vec_a, vec_b)
            cosine_sims.append(cos)

        result.mean_cosine_similarity = float(np.mean(cosine_sims))

        # --- L2 Distance (per sample, then average) ---
        l2_distances = []
        for i in range(len(flat_a)):
            vec_a = torch.tensor(flat_a[i], dtype=torch.float32)
            vec_b = torch.tensor(flat_b[i], dtype=torch.float32)
            l2 = l2_distance(vec_a, vec_b)
            l2_distances.append(l2)

        result.mean_l2_distance = float(np.mean(l2_distances))

        # --- KL Divergence (overall distribution) ---
        all_a = flat_a.flatten()
        all_b = flat_b.flatten()

        # Sample if too large
        if len(all_a) > 100000:
            indices = np.random.choice(len(all_a), 100000, replace=False)
            all_a = all_a[indices]
            all_b = all_b[indices]

        result.kl_divergence = kl_divergence(all_a, all_b)

        # --- Dead Neuron Analysis ---
        result.total_neurons = flat_a.shape[1]
        result.dead_neurons_a = int(np.sum(np.all(flat_a == 0, axis=0)))
        result.dead_neurons_b = int(np.sum(np.all(flat_b == 0, axis=0)))
        result.dead_neuron_change = (
            result.dead_neurons_b - result.dead_neurons_a
        )

        # --- Activation Entropy ---
        result.entropy_a = self._compute_activation_entropy(flat_a)
        result.entropy_b = self._compute_activation_entropy(flat_b)
        result.entropy_change = result.entropy_b - result.entropy_a

        # --- Drift Score ---
        result.activation_drift_score = self._compute_drift_score(result)
        result.drift_level = classify_drift(result.activation_drift_score)

        return result

    def _compute_activation_entropy(self, activations: np.ndarray) -> float:
        """
        Compute entropy of activation distribution.
        High entropy = diverse activations
        Low entropy = concentrated/dead activations
        """
        # Use histogram to estimate distribution
        flat = activations.flatten()

        if len(flat) == 0 or np.std(flat) == 0:
            return 0.0

        hist, _ = np.histogram(flat, bins=50, density=True)
        hist = hist + 1e-10  # avoid log(0)
        hist = hist / hist.sum()

        entropy = -np.sum(hist * np.log2(hist))
        return float(entropy)

    def _compute_drift_score(self, result: LayerActivationResult) -> float:
        """Compute normalized activation drift score (0 to 1)."""
        # Cosine distance
        cosine_distance = max(0.0, 1.0 - result.mean_cosine_similarity)

        # Normalize L2 by activation magnitude
        avg_magnitude = max(
            abs(result.mean_activation_a),
            abs(result.mean_activation_b),
            0.001,
        )
        normalized_l2 = min(
            result.mean_l2_distance / (avg_magnitude * result.total_neurons * 0.1 + 1),
            1.0,
        )

        # Normalize KL
        normalized_kl = min(result.kl_divergence / 5.0, 1.0)

        # Weighted combination
        score = (
            0.40 * cosine_distance +
            0.30 * normalized_l2 +
            0.30 * normalized_kl
        )

        return min(max(score, 0.0), 1.0)

    def _infer_layer_type(self, layer_name: str) -> str:
        """Try to infer layer type from name and model structure."""
        try:
            parts = layer_name.split(".")
            module = self.model_a
            for part in parts:
                if part.isdigit():
                    module = list(module.children())[int(part)]
                else:
                    module = getattr(module, part)
            return type(module).__name__
        except (AttributeError, IndexError):
            return "Unknown"