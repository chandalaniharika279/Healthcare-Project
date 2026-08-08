import pandas as pd

df = pd.read_csv("hospital_triage_o1_final.csv")

# ---- O2 Emergency Label ----
df["emergency"] = (df["esi"] <= 2).astype(int)

print(df["emergency"].value_counts())

df.to_csv("hospital_triage_02.csv", index=False)
print("O2 emergency labels created")
