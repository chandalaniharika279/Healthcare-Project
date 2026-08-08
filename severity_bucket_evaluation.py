import numpy as np
import pandas as pd

# Load model outputs
y_true = np.load("o1_true.npy")
y_pred = np.load("o1_pred.npy")

bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

true_bucket = pd.cut(y_true, bins=bins, labels=labels)
pred_bucket = pd.cut(y_pred, bins=bins, labels=labels)

df = pd.DataFrame({
    "True Severity": true_bucket,
    "Predicted Severity": pred_bucket
})

conf_matrix = pd.crosstab(
    df["True Severity"],
    df["Predicted Severity"]
)

print("\nSeverity Bucket Confusion Matrix:\n")
print(conf_matrix)

conf_matrix.to_csv("severity_bucket_confusion.csv")
