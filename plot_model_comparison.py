import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("model_comparison.csv", index_col=0)

plt.figure(figsize=(10,6))
plt.bar(df.index, df["MAE"])
plt.ylabel("Mean Absolute Error (↓ better)")
plt.title("Severity Prediction – Model Comparison")

plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
