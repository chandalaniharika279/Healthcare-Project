import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("baseline_model_results.csv")

plt.figure(figsize=(7,4))
plt.bar(df["Model"], df["MAE"])
plt.ylabel("Mean Absolute Error")
plt.title("Baseline Model Comparison (Severity Prediction)")
plt.grid(axis="y")

plt.tight_layout()
plt.show()
