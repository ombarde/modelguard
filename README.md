# 🛡️ ModelGuard
**Git diff for neural networks** — Compare, debug, and track ML model changes between versions.

[![PyPI version](https://badge.fury.io/py/modelguard-ai.svg)](https://pypi.org/project/modelguard-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

---

> **"My model accuracy dropped after retraining. I had no idea why. ModelGuard told me in 5 seconds."**

---

## The Problem

Every ML engineer has been here:

You retrain your model. Accuracy drops 3%. Data drift tools show nothing unusual. Feature distributions look fine. But something changed — you just can't see *what*, *where*, or *why*.

Most monitoring tools watch your **data**. Nobody is watching what happens **inside your model** — layer by layer, neuron by neuron — between versions.

**That's the gap ModelGuard fills.**

---

## What ModelGuard Does

```python
pip install modelguard-ai
```

```python
from modelguard import compare_models

report = compare_models(model_v1, model_v2, dataset=X_test,
                        class_names=["Stay", "Churn"],
                        feature_names=["credit_score", "age", "balance", ...])

report.summary()          # overall stability score
report.layer_drift()      # which weights changed
report.prediction_shift() # how outputs changed
report.activation_drift() # what neurons are doing differently
report.feature_sensitivity() # which features model relies on now
report.blame()            # which layer caused it
report.export("report.html")  # shareable HTML report
```

**That's it. 5 lines. Complete picture of everything that changed.**

---

## Real Example — Bank Churn Model

A bank retrains its churn prediction model after 6 months. Accuracy drops from 87% to 81%. Nobody knows why. ModelGuard finds it instantly:

```
╔══════════════════════════════════════╗
║  OVERALL STABILITY SCORE: 0.5821    ║
║  STATUS: 🔶 MODERATE DRIFT          ║
╚══════════════════════════════════════╝

Component Breakdown:
──────────────────────────────────────────
Weight Drift:     ███░░░░░░░ 0.3102  ⚠️  LOW DRIFT
Prediction Shift: ██████░░░░ 0.6341  🔴  HIGH DRIFT  ← 63% of predictions changed
Activation Drift: ████░░░░░░ 0.4892  🔶  MODERATE DRIFT
Feature Drift:    ███░░░░░░░ 0.3841  🔶  MODERATE DRIFT
```

**Blame Analysis — The Real Insight:**
```
🔍 BLAME ANALYSIS
Primary drift source: Layer 3 (Linear)
Drift score: 0.7802 (🔴 HIGH DRIFT)
Activation mean shifted: 0.0148 → -0.2898

📈 Model now relies MORE on: customer_complaints, num_transactions
📉 Model now relies LESS on: account_balance, credit_score, tenure_years
```

The model silently changed its decision logic over 6 months — shifting from financial signals to behavioral signals. No dashboard caught it. **ModelGuard found it in one run.**

---

## The 5 Analyzers

### 🔹 1. Weight Drift
Compares layer-by-layer weights using L2 norm, cosine similarity, and KL divergence. Shows exactly which layers physically changed and by how much.

```
+----------+------+---------+------------+--------+------------------+
| Layer    | Type | L2 Diff | Cosine Sim | Drift  | Status           |
+----------+------+---------+------------+--------+------------------+
| 0.weight | W    | 0.0000  | 1.0000     | 0.0000 | ✅ STABLE         |
| 2.weight | W    | 22.4072 | 0.1380     | 0.7703 | 🔴 HIGH DRIFT     |
| 4.bias   | B    | 0.8504  | 0.3601     | 0.8560 | 🚨 CRITICAL DRIFT |
+----------+------+---------+------------+--------+------------------+
```

### 🔹 2. Prediction Shift
Runs both models on the same dataset. Measures disagreement rate, confidence changes, and per-class probability shifts — so you know exactly how user-facing behavior changed.

### 🔹 3. Activation Drift ← The Unique One
**This is what makes ModelGuard different.**

Using forward hooks, ModelGuard captures intermediate layer activations from both model versions and compares them directly. Most tools compare inputs and outputs. ModelGuard compares what the neurons are doing in between — giving you internal visibility no other lightweight library provides.

```
+-------+--------+------------+---------+--------+------------------+
| Layer | Type   | Cosine Sim | L2 Dist | Drift  | Status           |
+-------+--------+------------+---------+--------+------------------+
| 0     | Linear | 1.0000     | 0.0000  | 0.0000 | ✅ STABLE         |
| 2     | Linear | -0.0079    | 9.2387  | 0.7802 | 🔴 HIGH DRIFT     |
| 4     | Linear | 0.2406     | 6.3760  | 0.7107 | 🔴 HIGH DRIFT     |
+-------+--------+------------+---------+--------+------------------+
```

### 🔹 4. Feature Influence Drift
Gradient-based attribution (Input × Gradient) compares which input features each model version relies on. Catches silent shifts in decision logic without needing SHAP or any external library.

```
📈 Top Gained: customer_complaints, num_transactions
📉 Top Lost:   credit_score, account_balance, tenure_years
🔄 Biggest rank change: complaints (+6 ranks), balance (-5 ranks)
```

### 🔹 5. Blame Analysis
Like `git blame` — but for model layers. Identifies the single layer most responsible for the behavioral change, with before/after activation statistics.

---

## Full API Reference

```python
from modelguard import compare_models

report = compare_models(
    model_a=model_v1,             # baseline model (torch.nn.Module)
    model_b=model_v2,             # new model version (torch.nn.Module)
    dataset=X_test,               # test dataset (torch.Tensor or np.ndarray)
    class_names=["Cat", "Dog"],   # optional — for classification
    feature_names=["age", "income", ...],  # optional — for feature drift
    skip_activations=False,       # set True for faster runs
    skip_features=False,          # set True for faster runs
    batch_size=64,
)

# Individual analyses
report.summary()              # overall stability score + key insights
report.layer_drift()          # weight-level comparison
report.prediction_shift()     # prediction behavior changes
report.activation_drift()     # internal representation changes
report.feature_sensitivity()  # feature importance changes
report.blame()                # which layer caused the problem
report.fingerprint()          # model stability fingerprint (MSF)

# Export
report.export("report.html")  # beautiful dark-themed HTML report
report.export("report.json")  # machine-readable
report.export("report.txt")   # plain text
```

---

## Model Stability Fingerprint

Every analysis generates a **Model Stability Fingerprint (MSF)** — a single numerical profile capturing model stability across all dimensions:

```
🔑 Model Stability Fingerprint (MSF):
─────────────────────────────────────────────
weight_drift              ██░░░░░░░░ 0.2852
prediction_drift          ███████░░░ 0.7189
disagreement_rate         ██████░░░░ 0.6800
confidence_change         █░░░░░░░░░ 0.1513
activation_drift          ████░░░░░░ 0.4753
feature_drift             ███░░░░░░░ 0.3842
overall_stability         ███░░░░░░░ 0.6131
─────────────────────────────────────────────
```

Archive this alongside every model version. Spot regressions instantly.

---

## How It Compares

| Feature                        | ModelGuard | Evidently AI | Deepchecks | W&B     |
|-------------------------------|:----------:|:------------:|:----------:|:-------:|
| Weight-level diff             | ✅          | ❌            | ❌          | ❌       |
| Activation drift (internals)  | ✅          | ❌            | ❌          | ❌       |
| Prediction shift              | ✅          | ✅            | ✅          | ✅       |
| Feature importance drift      | ✅          | ✅            | ✅          | ⚠️       |
| Blame analysis                | ✅          | ❌            | ❌          | ❌       |
| Unified 5-line API            | ✅          | ❌            | ❌          | ❌       |
| No platform / account needed  | ✅          | ✅            | ✅          | ❌       |
| pip install only              | ✅          | ✅            | ✅          | ❌       |

---

## Requirements

- Python 3.9+
- PyTorch 1.9+
- NumPy 1.21+
- SciPy 1.7+

---

## Use Cases

- **After retraining** — understand what actually changed between model versions
- **Production monitoring** — catch behavioral drift before it impacts users
- **CI/CD pipelines** — automated model validation before deployment *(CLI coming soon)*
- **Debugging** — find which layer is causing prediction instability
- **Research** — track how architecture changes affect internal representations
- **Auditing** — document model changes for compliance or stakeholder review

---

## Installation

```bash
pip install modelguard-ai
```

> **Note:** Install name is `modelguard-ai`. Import name is `modelguard`. The API stays clean.

---

## Roadmap

- [ ] CLI tool for CI/CD integration (`modelguard check --threshold 0.3`)
- [ ] TensorFlow / Keras support
- [ ] GitHub Actions example workflow
- [ ] Time-series model support
- [ ] Regression model support (currently classification-focused)
- [ ] Interactive HTML report with charts

---

## Contributing

Contributions are very welcome. Open an issue or submit a PR.

If you find a bug, have a feature request, or want to discuss the approach — open an issue and let's talk.

---

## Read More

- 📝 **Medium Article:** [I Built "Git Diff" for Neural Networks — Here's Why Every ML Engineer Needs It](https://medium.com/@bardeom6702/modelguard-ai-993d7baf0b93)
- 💬 **Reddit Discussion:** [r/MachineLearning](https://www.reddit.com/user/Shot-Personality7463/comments/1rla28i/i_built_a_git_diff_for_neural_networks_compares/)

---

## License

MIT License

---

*Built with ❤️ and a lot of frustration with unexplained model regressions.*

*If ModelGuard saved you debugging time, a ⭐ on GitHub is the best way to say thanks.*