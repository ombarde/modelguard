"""
ModelGuard Core API
The unified interface that ties all modules together.

Usage:
    from modelguard import compare_models

    report = compare_models(model_v1, model_v2, dataset=X_test)
    report.summary()
    report.layer_drift()
    report.prediction_shift()
    report.feature_sensitivity()
    report.blame()
    report.export("report.html")
"""

import torch
import numpy as np
from typing import List, Optional

from modelguard.utils import validate_model, validate_dataset
from modelguard.weight_drift import WeightDriftAnalyzer, WeightDriftReport
from modelguard.prediction_shift import PredictionShiftAnalyzer, PredictionShiftReport
from modelguard.activation_drift import ActivationDriftAnalyzer, ActivationDriftReport
from modelguard.feature_drift import FeatureDriftAnalyzer, FeatureDriftReport
from modelguard.report import DiffReport


def compare_models(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    dataset: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    layer_names: Optional[List[str]] = None,
    batch_size: int = 64,
    skip_activations: bool = False,
    skip_features: bool = False,
) -> "DiffReport":
    """
    Compare two PyTorch models and generate a comprehensive diff report.

    This is the main entry point for ModelGuard.

    Args:
        model_a: First model (baseline / old version)
        model_b: Second model (candidate / new version)
        dataset: Test dataset tensor (N, features). Required for
                 prediction, activation, and feature analysis.
        class_names: Optional list of class names for classification
        feature_names: Optional list of input feature names
        layer_names: Optional list of specific layers to analyze
        batch_size: Batch size for inference
        skip_activations: Skip activation drift analysis (faster)
        skip_features: Skip feature drift analysis (faster)

    Returns:
        DiffReport object with all analysis results

    Example:
        >>> from modelguard import compare_models
        >>> report = compare_models(model_v1, model_v2, dataset=X_test)
        >>> report.summary()
        >>> report.export("report.html")
    """
    # Validate inputs
    validate_model(model_a)
    validate_model(model_b)

    print("\n🔍 ModelGuard — Comparing models...")
    print("=" * 60)

    # --- 1. Weight Drift (always runs, no dataset needed) ---
    print("  [1/4] Analyzing weight drift...")
    weight_analyzer = WeightDriftAnalyzer(model_a, model_b)
    weight_report = weight_analyzer.analyze()
    print(f"         Done. Overall drift: {weight_report.overall_drift_score:.4f} "
          f"({weight_report.overall_drift_level})")

    # --- 2. Prediction Shift (needs dataset) ---
    prediction_report = None
    if dataset is not None:
        dataset = validate_dataset(dataset)
        print("  [2/4] Analyzing prediction shift...")
        pred_analyzer = PredictionShiftAnalyzer(
            model_a, model_b, class_names=class_names
        )
        prediction_report = pred_analyzer.analyze(dataset, batch_size=batch_size)
        print(f"         Done. Disagreement rate: {prediction_report.disagreement_rate:.2%} "
              f"({prediction_report.prediction_drift_level})")
    else:
        print("  [2/4] Skipping prediction shift (no dataset provided)")

    # --- 3. Activation Drift (needs dataset) ---
    activation_report = None
    if dataset is not None and not skip_activations:
        print("  [3/4] Analyzing activation drift...")
        act_analyzer = ActivationDriftAnalyzer(
            model_a, model_b, layer_names=layer_names
        )
        activation_report = act_analyzer.analyze(dataset, batch_size=batch_size)
        print(f"         Done. Overall activation drift: "
              f"{activation_report.overall_activation_drift:.4f} "
              f"({activation_report.overall_drift_level})")
    else:
        reason = "no dataset" if dataset is None else "skipped"
        print(f"  [3/4] Skipping activation drift ({reason})")

    # --- 4. Feature Drift (needs dataset) ---
    feature_report = None
    if dataset is not None and not skip_features:
        print("  [4/4] Analyzing feature drift...")
        feat_analyzer = FeatureDriftAnalyzer(
            model_a, model_b, feature_names=feature_names
        )
        feature_report = feat_analyzer.analyze(dataset, batch_size=batch_size)
        print(f"         Done. Feature drift: {feature_report.overall_feature_drift:.4f} "
              f"({feature_report.overall_drift_level})")
    else:
        reason = "no dataset" if dataset is None else "skipped"
        print(f"  [4/4] Skipping feature drift ({reason})")

    print("=" * 60)
    print("✅ Analysis complete!\n")

    # Build unified report
    report = DiffReport(
        weight_report=weight_report,
        prediction_report=prediction_report,
        activation_report=activation_report,
        feature_report=feature_report,
        model_a_name="Model A (baseline)",
        model_b_name="Model B (candidate)",
    )

    return report