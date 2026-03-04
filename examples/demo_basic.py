"""
ModelGuard Demo — Full Pipeline
Shows the clean 5-line API in action.
"""

import torch
import torch.nn as nn
from modelguard import compare_models


# --- Create two model versions ---

def create_model():
    return nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 3),
    )


# Model V1 — baseline
model_v1 = create_model()

# Model V2 — "retrained" (simulate by modifying weights)
model_v2 = create_model()
model_v2.load_state_dict(model_v1.state_dict())

with torch.no_grad():
    # Simulate training drift — middle layers changed
    model_v2[2].weight.add_(torch.randn_like(model_v2[2].weight) * 0.5)
    model_v2[4].weight.add_(torch.randn_like(model_v2[4].weight) * 0.3)
    model_v2[4].bias.add_(torch.randn_like(model_v2[4].bias) * 0.2)


# --- Create test dataset ---
X_test = torch.randn(200, 10)


# --- THE CLEAN 5-LINE API ---

# 1. Compare models
report = compare_models(
    model_v1, model_v2,
    dataset=X_test,
    class_names=["Cat", "Dog", "Bird"],
    feature_names=[
        "age", "income", "score", "tenure", "balance",
        "products", "active", "salary", "credit", "region"
    ],
)

# 2. Overall summary
report.summary()

# 3. Detailed reports
report.layer_drift()
report.prediction_shift()
report.activation_drift()
report.feature_sensitivity()

# 4. Blame analysis
report.blame()

# 5. Model Stability Fingerprint
report.fingerprint()

# 6. Export reports
report.export("report.html")
report.export("report.json")
report.export("report.txt")

print("\n🎉 Demo complete! Check report.html in your browser.")