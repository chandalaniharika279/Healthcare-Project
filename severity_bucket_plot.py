import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("hospital_triage_o2.csv")

bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

df["severity_bucket"] = pd.cut(
    df["severity_score"], bins=bins, labels=labels
)

bucket_counts = df["severity_bucket"].value_counts().sort_index()

plt.figure()
bucket_counts.plot(kind="bar")
plt.title("Severity Distribution Across Buckets")
plt.xlabel("Severity Level")
plt.ylabel("Number of Cases")
plt.show()
