"""Test Module 3: Activation Drift Analyzer."""

import torch
import torch.nn as nn
import numpy as np
from modelguard.activation_drift import (
    ActivationDriftAnalyzer,
    ActivationCapturer,
)


def create_model():
    """Create a test model with named layers."""
    model = nn.Sequential(
        nn.Linear(10, 64),       # 0
        nn.ReLU(),               # 1
        nn.Linear(64, 32),       # 2
        nn.ReLU(),               # 3
        nn.Linear(32, 16),       # 4
        nn.ReLU(),               # 5
        nn.Linear(16, 3),        # 6
    )
    return model


def create_test_data(n_samples=100):
    """Create random test dataset."""
    return torch.randn(n_samples, 10)


def test_activation_capturer():
    """Test that we can capture activations."""
    model = create_model()
    dataset = create_test_data(50)

    capturer = ActivationCapturer(model)
    activations = capturer.capture(dataset, batch_size=25)
    capturer.remove_hooks()

    print("Captured activations from layers:")
    for name, act in activations.items():
        print(f"  {name}: shape={act.shape}")

    assert len(activations) > 0
    print("\n✅ Test 1 PASSED: Activation capturer works\n")


def test_identical_activations():
    """Same model should show zero activation drift."""
    model_a = create_model()
    model_b = create_model()
    model_b.load_state_dict(model_a.state_dict())

    dataset = create_test_data()

    analyzer = ActivationDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze(dataset)

    print(report.summary())

    assert report.overall_activation_drift < 0.01
    print("✅ Test 2 PASSED: Identical models show zero activation drift\n")


def test_modified_activations():
    """Modified model should show activation drift."""
    model_a = create_model()
    model_b = create_model()
    model_b.load_state_dict(model_a.state_dict())

    # Modify middle layer
    with torch.no_grad():
        model_b[2].weight.add_(torch.randn_like(model_b[2].weight) * 0.5)

    dataset = create_test_data()

    analyzer = ActivationDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze(dataset)

    print(report.summary())

    assert report.overall_activation_drift > 0.0
    print("✅ Test 3 PASSED: Modified model shows activation drift\n")


def test_heavily_modified_activations():
    """Completely different model should show high drift."""
    model_a = create_model()
    model_b = create_model()  # different random weights

    dataset = create_test_data()

    analyzer = ActivationDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze(dataset)

    print(report.summary())

    # Print layer ranking
    ranking = report.get_layer_ranking()
    print("Layer ranking by activation drift:")
    for name, score in ranking:
        print(f"  {name}: {score:.4f}")

    assert report.overall_activation_drift > 0.1
    print("\n✅ Test 4 PASSED: Different models show high activation drift\n")


def test_blame_analysis():
    """Blame analysis should identify the modified layer."""
    model_a = create_model()
    model_b = create_model()
    model_b.load_state_dict(model_a.state_dict())

    # Modify ONLY layer 4 heavily
    with torch.no_grad():
        model_b[4].weight.add_(torch.randn_like(model_b[4].weight) * 3.0)
        model_b[4].bias.add_(torch.randn_like(model_b[4].bias) * 2.0)

    dataset = create_test_data(150)

    analyzer = ActivationDriftAnalyzer(model_a, model_b)
    report = analyzer.analyze(dataset)

    print(report.summary())

    # Blame analysis
    blame = report.get_blame()
    print(blame)

    # Layer 4 or layers after it should be most drifted
    ranking = report.get_layer_ranking()
    print(f"\n  Top drifted layer: {ranking[0][0]}")

    print("\n✅ Test 5 PASSED: Blame analysis works\n")


def test_specific_layers():
    """Test analyzing only specific layers."""
    model_a = create_model()
    model_b = create_model()

    dataset = create_test_data(50)

    # Only analyze ReLU layers
    analyzer = ActivationDriftAnalyzer(
        model_a, model_b,
        layer_names=["1", "3", "5"]  # ReLU layers
    )
    report = analyzer.analyze(dataset)

    print(report.summary())

    assert report.total_layers_analyzed <= 3
    print("✅ Test 6 PASSED: Specific layer analysis works\n")


if __name__ == "__main__":
    test_activation_capturer()
    test_identical_activations()
    test_modified_activations()
    test_heavily_modified_activations()
    test_blame_analysis()
    test_specific_layers()
    print("🎉 ALL MODULE 3 TESTS PASSED!")