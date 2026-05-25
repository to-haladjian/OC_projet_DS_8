API_VERSION = "1.0.0"

# Optimal classification threshold, calibrated on the cost function
# (FN_COST=10, FP_COST=1) when retraining the LightGBM in scripts/train_export.py.
OPTIMAL_THRESHOLD = 0.0850
