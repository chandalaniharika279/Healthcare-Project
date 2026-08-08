import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load results (same file you already used)
df = pd.read_csv("hospital_triage_o2.csv")

# Buckets
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

df["true_bucket"] = pd.cut(df["severity_score"], bins=bins, labels=labels)
df["pred_bucket"] = pd.cut(df["severity_score"], bins=bins, labels=labels)  
# ⬆️ replace with model prediction if available

# Confusion matrix
cm = pd.crosstab(df["true_bucket"], df["pred_bucket"])

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Severity Bucket Confusion Matrix")
plt.xlabel("Predicted Severity Bucket")
plt.ylabel("True Severity Bucket")
plt.tight_layout()
plt.show()
