# 🛡️ ModelGuard

**Git diff for neural networks** — Compare, debug, and track ML model changes.

[![PyPI version](https://badge.fury.io/py/modelguard.svg)](https://pypi.org/project/modelguard-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

ML engineers constantly struggle with:

- *"Why did my model accuracy drop after retraining?"*
- *"Which layer caused instability?"*
- *"Why are production predictions different?"*
- *"Which features is the model relying on now?"*

**Existing tools** handle data drift OR model monitoring separately.  
**ModelGuard** unifies weight analysis, activation diffing, prediction shift, and feature drift into **one pip-installable library**.

---

## Installation

```bash
pip install modelguard-ai