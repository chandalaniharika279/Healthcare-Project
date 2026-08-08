import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# LOAD BERT OUTPUTS
# =====================================================

y_true = np.load("o1_true.npy")
y_pred = np.load("o1_pred.npy")

# =====================================================
# CLINICAL BUCKETS
# =====================================================

def bucket(scores):
    return np.digitize(
        scores,
        bins=[0.25,0.50,0.75],
        right=True
    )

true_bucket = bucket(y_true)
pred_bucket = bucket(y_pred)

# =====================================================
# CALCULATE METRICS
# =====================================================

mse = mean_squared_error(y_true,y_pred)
mae = mean_absolute_error(y_true,y_pred)
r2  = r2_score(y_true,y_pred)

acc = accuracy_score(
    true_bucket,
    pred_bucket
)

tol = np.mean(
    np.abs(true_bucket-pred_bucket)<=1
)

precision = precision_score(
    true_bucket,
    pred_bucket,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_bucket,
    pred_bucket,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_bucket,
    pred_bucket,
    average="weighted",
    zero_division=0
)

# =====================================================
# LOAD BASELINE TABLE
# =====================================================

baseline = pd.read_csv(
    "baseline_model_results.csv"
)

# =====================================================
# ADD PROPOSED MODEL
# =====================================================

bert_row = pd.DataFrame([{

    "Model":"BERT (Proposed)",

    "MSE":mse,

    "MAE":mae,

    "R2":r2,

    "Accuracy":acc,

    "ToleranceAcc":tol,

    "Precision":precision,

    "Recall":recall,

    "F1":f1

}])

final = pd.concat(
    [baseline,bert_row],
    ignore_index=True
)

# =====================================================
# SAVE
# =====================================================

final.to_csv(
    "final_model_results.csv",
    index=False
)

print("\n======================================")
print("FINAL MODEL COMPARISON")
print("======================================")

print(final)

print("\nSaved as final_model_results.csv")

# =====================================================
# BEST MODEL
# =====================================================

print("\nBest MAE")
print(final.loc[
    final["MAE"].idxmin()
])

print("\nBest Recall")
print(final.loc[
    final["Recall"].idxmax()
])

print("\nBest F1")
print(final.loc[
    final["F1"].idxmax()
])