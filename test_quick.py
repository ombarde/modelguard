# test_quick.py — put this in E:\modelguard\

import sys
print("Testing imports one by one...\n")

try:
    from modelguard.utils import validate_model
    print("✅ utils.py OK")
except Exception as e:
    print(f"❌ utils.py FAILED: {e}")

try:
    from modelguard.weight_drift import WeightDriftAnalyzer
    print("✅ weight_drift.py OK")
except Exception as e:
    print(f"❌ weight_drift.py FAILED: {e}")

try:
    from modelguard.prediction_shift import PredictionShiftAnalyzer
    print("✅ prediction_shift.py OK")
except Exception as e:
    print(f"❌ prediction_shift.py FAILED: {e}")

try:
    from modelguard.activation_drift import ActivationDriftAnalyzer
    print("✅ activation_drift.py OK")
except Exception as e:
    print(f"❌ activation_drift.py FAILED: {e}")

try:
    from modelguard.feature_drift import FeatureDriftAnalyzer
    print("✅ feature_drift.py OK")
except Exception as e:
    print(f"❌ feature_drift.py FAILED: {e}")

try:
    from modelguard.report import DiffReport
    print("✅ report.py OK")
except Exception as e:
    print(f"❌ report.py FAILED: {e}")

try:
    from modelguard.core import compare_models
    print("✅ core.py OK")
except Exception as e:
    print(f"❌ core.py FAILED: {e}")

print("\nDone!")