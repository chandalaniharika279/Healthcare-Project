# severity_bucket_analysis.py
import numpy as np
import pandas as pd

# Load saved predictions
y_true = np.load("o1_true.npy")
y_pred = np.load("o1_pred.npy")

# Define severity bins
bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

# Bucketize true and predicted severity
true_bins = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)
pred_bins = pd.cut(y_pred, bins=bins, labels=labels, include_lowest=True)

# Create comparison table
df = pd.DataFrame({
    "True Severity Bucket": true_bins,
    "Predicted Severity Bucket": pred_bins
})

# Count matches per bucket
bucket_analysis = (
    df.groupby(["True Severity Bucket", "Predicted Severity Bucket"])
      .size()
      .unstack(fill_value=0)
)

print("\n=== Severity Bucket Confusion Matrix ===\n")
print(bucket_analysis)

# Save for paper / plotting
bucket_analysis.to_csv("severity_bucket_analysis.csv")
