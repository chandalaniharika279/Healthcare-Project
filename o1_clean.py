import pandas as pd
import numpy as np

df = pd.read_csv("hospital_triage_o1.csv")

# 1. Drop missing ESI
df = df.dropna(subset=["esi"])

# 2. Ensure severity_score is numeric
df["severity_score"] = pd.to_numeric(df["severity_score"], errors="coerce")

# 3. Drop NaN severity
df = df.dropna(subset=["severity_score"])

# 4. Ensure symptom_text is valid
df["symptom_text"] = df["symptom_text"].astype(str)
df = df[df["symptom_text"].str.strip() != ""]
df = df[df["symptom_text"] != "nan"]

print("After strict cleaning rows:", len(df))
print(df[["esi", "severity_score"]].head())

df.to_csv("hospital_triage_o1_final.csv", index=False)
print("O1 final clean dataset saved")
