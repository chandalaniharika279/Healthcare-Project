import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# ---------- LOAD DATA ----------
y_true = np.load("o1_true.npy")


models = {
    "BERT": np.load("bert_pred.npy"),
    "LogReg": np.load("lr_pred.npy"),
    "RandomForest": np.load("rf_pred.npy"),
    "SVM": np.load("svm_pred.npy"),
    "LSTM": np.load("lstm_pred.npy"),
    "CNN": np.load("cnn_pred.npy"),
}

# ---------- BUCKETS ----------
bins = [(0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.0)]
labels = ["20%","40%","60%","80%","100%"]

rows = []

for model_name, pred in models.items():
    for (low, high), label in zip(bins, labels):
        idx = (y_true >= low) & (y_true < high)

        if idx.sum() == 0:
            continue

        mae = mean_absolute_error(y_true[idx], pred[idx])
        rows.append([model_name, label, mae])

df = pd.DataFrame(rows, columns=["Model", "Bucket", "MAE"])
df.to_csv("bucket_results.csv", index=False)

print("✅ Bucket-wise comparison saved to bucket_results.csv")
print(df)
