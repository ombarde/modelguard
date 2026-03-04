"""
Module 2: Prediction Shift Engine
Compares prediction behavior between two models on the same dataset.

Detects:
- Prediction probability shifts
- Confidence changes
- Class-wise behavior changes
- Disagreement rate
- Calibration drift
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from modelguard.utils import validate_model, validate_dataset, classify_drift


@dataclass
class ClassShiftResult:
    """Prediction shift for a single class."""
    class_id: int
    class_name: str
    avg_prob_model_a: float = 0.0
    avg_prob_model_b: float = 0.0
    prob_change: float = 0.0          # absolute change
    count_model_a: int = 0            # how many predicted this class
    count_model_b: int = 0
    count_change: int = 0


@dataclass
class PredictionShiftReport:
    """Complete prediction shift analysis report."""
    total_samples: int = 0
    disagreement_count: int = 0
    disagreement_rate: float = 0.0
    avg_confidence_model_a: float = 0.0
    avg_confidence_model_b: float = 0.0
    confidence_change: float = 0.0
    avg_probability_shift: float = 0.0    # mean absolute diff across all probs
    max_probability_shift: float = 0.0
    prediction_drift_score: float = 0.0   # normalized 0-1
    prediction_drift_level: str = ""
    class_shifts: List[ClassShiftResult] = field(default_factory=list)
    flipped_indices: List[int] = field(default_factory=list)

    # Raw data for further analysis
    probs_model_a: Optional[np.ndarray] = None
    probs_model_b: Optional[np.ndarray] = None
    preds_model_a: Optional[np.ndarray] = None
    preds_model_b: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Print clean summary of prediction shift."""
        lines = []
        lines.append("\n" + "=" * 65)
        lines.append("  🔮 PREDICTION SHIFT ANALYSIS REPORT")
        lines.append("=" * 65)

        lines.append(f"\n  Total Samples Compared: {self.total_samples}")
        lines.append(f"  Predictions Disagreed:  {self.disagreement_count} / {self.total_samples}")
        lines.append(f"  Disagreement Rate:      {self.disagreement_rate:.2%}")

        lines.append(f"\n  Avg Confidence (Model A): {self.avg_confidence_model_a:.4f}")
        lines.append(f"  Avg Confidence (Model B): {self.avg_confidence_model_b:.4f}")
        lines.append(f"  Confidence Change:        {self.confidence_change:+.4f}")

        lines.append(f"\n  Avg Probability Shift:    {self.avg_probability_shift:.4f}")
        lines.append(f"  Max Probability Shift:    {self.max_probability_shift:.4f}")

        lines.append(f"\n  Prediction Drift Score:   {self.prediction_drift_score:.4f}")
        lines.append(f"  Prediction Drift Level:   {self.prediction_drift_level}")

        # Class-wise breakdown
        if self.class_shifts:
            lines.append("\n" + "-" * 65)
            lines.append("  Class-wise Breakdown:")
            lines.append("-" * 65)

            lines.append(
                f"  {'Class':<10} {'Prob A':>8} {'Prob B':>8} "
                f"{'Change':>8} {'Count A':>9} {'Count B':>9}"
            )
            lines.append("  " + "-" * 55)

            for cs in self.class_shifts:
                change_symbol = "📈" if cs.prob_change > 0.01 else (
                    "📉" if cs.prob_change < -0.01 else "➡️"
                )
                lines.append(
                    f"  {cs.class_name:<10} {cs.avg_prob_model_a:>8.4f} "
                    f"{cs.avg_prob_model_b:>8.4f} {cs.prob_change:>+8.4f} "
                    f"{cs.count_model_a:>9} {cs.count_model_b:>9} {change_symbol}"
                )

        # Flipped predictions warning
        if self.flipped_indices:
            num_show = min(10, len(self.flipped_indices))
            lines.append(f"\n  ⚠️ {len(self.flipped_indices)} samples changed prediction")
            lines.append(f"  First {num_show} flipped indices: {self.flipped_indices[:num_show]}")

        lines.append("=" * 65 + "\n")
        return "\n".join(lines)

    def get_flipped_samples(self) -> List[Dict]:
        """Get details of samples where prediction changed."""
        flipped = []
        if self.probs_model_a is None or self.probs_model_b is None:
            return flipped

        for idx in self.flipped_indices:
            flipped.append({
                "index": idx,
                "pred_model_a": int(self.preds_model_a[idx]),
                "pred_model_b": int(self.preds_model_b[idx]),
                "confidence_a": float(self.probs_model_a[idx].max()),
                "confidence_b": float(self.probs_model_b[idx].max()),
                "probs_a": self.probs_model_a[idx].tolist(),
                "probs_b": self.probs_model_b[idx].tolist(),
            })

        return flipped


class PredictionShiftAnalyzer:
    """
    Compares prediction behavior of two models on the same dataset.

    Supports:
    - Classification models (softmax output)
    - Models that output raw logits (auto-applies softmax)

    Metrics:
    - Disagreement rate
    - Confidence shift
    - Per-class probability changes
    - Sample-level prediction flips
    """

    def __init__(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        class_names: Optional[List[str]] = None,
    ):
        validate_model(model_a)
        validate_model(model_b)

        self.model_a = model_a
        self.model_b = model_b
        self.class_names = class_names

    def analyze(
        self,
        dataset: torch.Tensor,
        batch_size: int = 64,
    ) -> PredictionShiftReport:
        """
        Run prediction shift analysis.

        Args:
            dataset: Input tensor (N, ...) — N samples
            batch_size: Batch size for inference

        Returns:
            PredictionShiftReport with all metrics
        """
        dataset = validate_dataset(dataset)

        # Get predictions from both models
        probs_a = self._get_predictions(self.model_a, dataset, batch_size)
        probs_b = self._get_predictions(self.model_b, dataset, batch_size)

        # Build report
        report = self._compute_metrics(probs_a, probs_b)

        return report

    def _get_predictions(
        self,
        model: torch.nn.Module,
        dataset: torch.Tensor,
        batch_size: int,
    ) -> np.ndarray:
        """Run inference and get probability outputs."""
        model.eval()
        all_probs = []

        with torch.no_grad():
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]

                # Handle device
                device = next(model.parameters()).device
                batch = batch.to(device)

                output = model(batch)

                # Convert logits to probabilities if needed
                probs = self._to_probabilities(output)

                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def _to_probabilities(self, output: torch.Tensor) -> torch.Tensor:
        """Convert model output to probabilities."""
        # If already probabilities (sums to ~1)
        if output.dim() >= 2:
            row_sums = output.sum(dim=-1)
            if torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01):
                return output

        # If single value (binary classification sigmoid)
        if output.dim() == 1 or (output.dim() == 2 and output.shape[1] == 1):
            sigmoid_out = torch.sigmoid(output.flatten())
            return torch.stack([1 - sigmoid_out, sigmoid_out], dim=1)

        # Multi-class: apply softmax
        return torch.softmax(output, dim=-1)

    def _compute_metrics(
        self,
        probs_a: np.ndarray,
        probs_b: np.ndarray,
    ) -> PredictionShiftReport:
        """Compute all prediction shift metrics."""
        report = PredictionShiftReport()

        n_samples = probs_a.shape[0]
        n_classes = probs_a.shape[1]
        report.total_samples = n_samples

        # Store raw data
        report.probs_model_a = probs_a
        report.probs_model_b = probs_b

        # Get predicted classes
        preds_a = np.argmax(probs_a, axis=1)
        preds_b = np.argmax(probs_b, axis=1)
        report.preds_model_a = preds_a
        report.preds_model_b = preds_b

        # --- Disagreement ---
        disagreements = preds_a != preds_b
        report.disagreement_count = int(np.sum(disagreements))
        report.disagreement_rate = report.disagreement_count / n_samples
        report.flipped_indices = list(np.where(disagreements)[0])

        # --- Confidence ---
        confidence_a = np.max(probs_a, axis=1)
        confidence_b = np.max(probs_b, axis=1)
        report.avg_confidence_model_a = float(np.mean(confidence_a))
        report.avg_confidence_model_b = float(np.mean(confidence_b))
        report.confidence_change = (
            report.avg_confidence_model_b - report.avg_confidence_model_a
        )

        # --- Probability Shift ---
        prob_diffs = np.abs(probs_a - probs_b)
        report.avg_probability_shift = float(np.mean(prob_diffs))
        report.max_probability_shift = float(np.max(prob_diffs))

        # --- Class-wise Analysis ---
        class_names = self.class_names or [
            f"Class_{i}" for i in range(n_classes)
        ]

        for c in range(n_classes):
            cs = ClassShiftResult(
                class_id=c,
                class_name=class_names[c] if c < len(class_names) else f"Class_{c}",
                avg_prob_model_a=float(np.mean(probs_a[:, c])),
                avg_prob_model_b=float(np.mean(probs_b[:, c])),
                count_model_a=int(np.sum(preds_a == c)),
                count_model_b=int(np.sum(preds_b == c)),
            )
            cs.prob_change = cs.avg_prob_model_b - cs.avg_prob_model_a
            cs.count_change = cs.count_model_b - cs.count_model_a

            report.class_shifts.append(cs)

        # --- Overall Drift Score ---
        report.prediction_drift_score = self._compute_prediction_drift_score(
            report.disagreement_rate,
            report.avg_probability_shift,
            abs(report.confidence_change),
        )
        report.prediction_drift_level = classify_drift(
            report.prediction_drift_score
        )

        return report

    def _compute_prediction_drift_score(
        self,
        disagreement_rate: float,
        avg_prob_shift: float,
        confidence_change: float,
    ) -> float:
        """
        Compute normalized prediction drift score (0 to 1).

        Combines:
        - Disagreement rate (strongest signal)
        - Average probability shift
        - Confidence change
        """
        # Normalize each component
        norm_disagreement = min(disagreement_rate / 0.5, 1.0)
        norm_prob_shift = min(avg_prob_shift / 0.3, 1.0)
        norm_confidence = min(confidence_change / 0.3, 1.0)

        # Weighted combination
        score = (
            0.5 * norm_disagreement +
            0.3 * norm_prob_shift +
            0.2 * norm_confidence
        )

        return min(max(score, 0.0), 1.0)