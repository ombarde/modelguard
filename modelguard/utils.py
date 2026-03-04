"""
Utility functions used across all modules.
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


def validate_model(model):
    """Check if input is a valid PyTorch model."""
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Expected torch.nn.Module, got {type(model).__name__}. "
            f"ModelGuard currently supports PyTorch models only."
        )


def validate_dataset(dataset):
    """Check if dataset is a valid tensor."""
    if isinstance(dataset, np.ndarray):
        return torch.tensor(dataset, dtype=torch.float32)
    elif isinstance(dataset, torch.Tensor):
        return dataset
    else:
        raise TypeError(
            f"Expected numpy array or torch.Tensor, got {type(dataset).__name__}"
        )


def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    vec_a_flat = vec_a.detach().flatten().float()
    vec_b_flat = vec_b.detach().flatten().float()

    dot_product = torch.dot(vec_a_flat, vec_b_flat)
    norm_a = torch.norm(vec_a_flat)
    norm_b = torch.norm(vec_b_flat)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return (dot_product / (norm_a * norm_b)).item()


def l2_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute L2 (Euclidean) distance between two tensors."""
    vec_a_flat = vec_a.detach().flatten().float()
    vec_b_flat = vec_b.detach().flatten().float()
    return torch.norm(vec_a_flat - vec_b_flat).item()


def kl_divergence(
    dist_a: np.ndarray,
    dist_b: np.ndarray,
    bins: int = 100
) -> float:
    """
    Compute KL divergence between two distributions.
    Converts raw values to histograms first.
    """
    # Create histograms from raw values
    min_val = min(dist_a.min(), dist_b.min())
    max_val = max(dist_a.max(), dist_b.max())

    hist_a, bin_edges = np.histogram(
        dist_a, bins=bins, range=(min_val, max_val), density=True
    )
    hist_b, _ = np.histogram(
        dist_b, bins=bins, range=(min_val, max_val), density=True
    )

    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    hist_a = hist_a + epsilon
    hist_b = hist_b + epsilon

    # Normalize to make valid probability distributions
    hist_a = hist_a / hist_a.sum()
    hist_b = hist_b / hist_b.sum()

    return float(stats.entropy(hist_a, hist_b))


def mean_absolute_change(
    vec_a: torch.Tensor,
    vec_b: torch.Tensor
) -> float:
    """Compute mean absolute difference between two tensors."""
    vec_a_flat = vec_a.detach().flatten().float()
    vec_b_flat = vec_b.detach().flatten().float()
    return torch.mean(torch.abs(vec_a_flat - vec_b_flat)).item()


def classify_drift(score: float) -> str:
    """Classify drift level based on score."""
    if score < 0.1:
        return "✅ STABLE"
    elif score < 0.3:
        return "⚠️ LOW DRIFT"
    elif score < 0.6:
        return "🔶 MODERATE DRIFT"
    elif score < 0.8:
        return "🔴 HIGH DRIFT"
    else:
        return "🚨 CRITICAL DRIFT"


def get_layer_type(layer_name: str, model: torch.nn.Module) -> str:
    """Get the type of a layer from its parameter name."""
    parts = layer_name.replace(".weight", "").replace(".bias", "").split(".")

    module = model
    try:
        for part in parts:
            if part.isdigit():
                module = list(module.children())[int(part)]
            else:
                module = getattr(module, part)
        return type(module).__name__
    except (AttributeError, IndexError):
        return "Unknown"


def format_table(
    headers: List[str],
    rows: List[List[str]],
    col_widths: Optional[List[int]] = None
) -> str:
    """Format data as a clean ASCII table."""
    if col_widths is None:
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(header)
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(max_width + 2)

    # Build separator
    separator = "+" + "+".join("-" * w for w in col_widths) + "+"

    # Build header
    header_str = "|"
    for i, header in enumerate(headers):
        header_str += f" {header:<{col_widths[i]-2}} |"

    # Build rows
    row_strings = []
    for row in rows:
        row_str = "|"
        for i, cell in enumerate(row):
            row_str += f" {str(cell):<{col_widths[i]-2}} |"
        row_strings.append(row_str)

    # Combine
    lines = [separator, header_str, separator]
    lines.extend(row_strings)
    lines.append(separator)

    return "\n".join(lines)