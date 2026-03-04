"""
ModelGuard — Git Diff for Neural Networks
Compare, debug, and track ML model changes.
"""

__version__ = "0.1.0"
__author__ = "Om Barde"

from modelguard.core import compare_models
from modelguard.report import DiffReport
from modelguard.weight_drift import WeightDriftAnalyzer
from modelguard.prediction_shift import PredictionShiftAnalyzer
from modelguard.activation_drift import ActivationDriftAnalyzer
from modelguard.feature_drift import FeatureDriftAnalyzer

__all__ = [
    "compare_models",
    "DiffReport",
    "WeightDriftAnalyzer",
    "PredictionShiftAnalyzer",
    "ActivationDriftAnalyzer",
    "FeatureDriftAnalyzer",
]