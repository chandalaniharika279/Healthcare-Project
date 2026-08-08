# =====================================================
# FULL SEVERITY-AWARE EVALUATION PIPELINE (FINAL)
# =====================================================

import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, recall_score, accuracy_score

from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# 0. SETUP
# =====================================================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Device:", device)

EMERGENCY_THRESHOLD = 0.7

def to_emergency(y):
    return (y >= EMERGENCY_THRESHOLD).astype(int)

# =====================================================
# 1. LOAD + SUBSAMPLE DATA
# =====================================================
print("\n📥 Loading data...")
train_df = pd.read_csv("D:/Nandini/tra_in_o1.csv")
test_df  = pd.read_csv("D:/Nandini/te_st_o1.csv")

TEXT_COL = "symptom_text"
TARGET = "severity_score"

train_df = train_df.dropna(subset=[TEXT_COL, TARGET])
test_df  = test_df.dropna(subset=[TEXT_COL, TARGET])

train_df = train_df
test_df  = test_df

X_train = train_df[TEXT_COL].astype(str)
X_test  = test_df[TEXT_COL].astype(str)
y_train = train_df[TARGET].values
y_test  = test_df[TARGET].values

# Storage
all_preds = {}
all_true  = {}

# =====================================================
# 2. TF-IDF BASELINE MODELS
# =====================================================
print("\n🔤 TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=3000, min_df=20, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ---- Linear Regression ----
print("▶ Linear Regression")
lin = LinearRegression(n_jobs=-1)
lin.fit(X_train_tfidf, y_train)
pred = lin.predict(X_test_tfidf)
all_preds["Linear Regression"] = pred
all_true["Linear Regression"] = y_test

# ---- Logistic Regression ----
print("▶ Logistic Regression")
y_train_cls = np.clip((y_train * 5).round().astype(int), 1, 5)
log = LogisticRegression(max_iter=500, n_jobs=-1)
log.fit(X_train_tfidf, y_train_cls)
pred = log.predict(X_test_tfidf) / 5.0
all_preds["Logistic Regression"] = pred
all_true["Logistic Regression"] = y_test

# ---- Random Forest ----
print("▶ Random Forest")
rf = RandomForestRegressor(n_estimators=30, max_depth=12, n_jobs=-1, random_state=42)
rf.fit(X_train_tfidf, y_train)
pred = rf.predict(X_test_tfidf[:8000])
all_preds["Random Forest"] = pred
all_true["Random Forest"] = y_test[:8000]

# ---- SVM ----
print("▶ SVM")
svm = LinearSVR()
svm.fit(X_train_tfidf, y_train)
pred = svm.predict(X_test_tfidf[:2000])
all_preds["SVM"] = pred
all_true["SVM"] = y_test[:2000]

# =====================================================
# 3. CNN / LSTM DATA PREP
# =====================================================
print("\n📚 Preparing sequence data...")
MAX_VOCAB = 8000
MAX_LEN = 40

counter = Counter(" ".join(X_train[:20000]).split())
vocab = {"<pad>": 0, "<unk>": 1}
for w, _ in counter.most_common(MAX_VOCAB - 2):
    vocab[w] = len(vocab)

def encode(text):
    ids = [vocab.get(t, 1) for t in text.split()][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

class SeqDS(Dataset):
    def __init__(self, texts, labels):
        self.X = [encode(t) for t in texts]
        self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.tensor(self.X[i]), torch.tensor(self.y[i], dtype=torch.float)

train_seq = SeqDS(X_train, y_train)
test_seq  = SeqDS(X_test, y_test)

train_loader = DataLoader(train_seq, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_seq, batch_size=64)

# =====================================================
# 4. CNN
# =====================================================
print("\n🧠 Training CNN")
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), 64)
        self.conv = nn.Conv1d(64, 64, 3)
        self.fc = nn.Linear(64, 1)
    def forward(self, x):
        x = self.emb(x).permute(0,2,1)
        x = torch.relu(self.conv(x))
        x = torch.max(x, 2)[0]
        return torch.sigmoid(self.fc(x)).squeeze()

cnn = CNN().to(device)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(2):
    print(f"  CNN Epoch {epoch+1}/2")
    for x,y in train_loader:
        opt.zero_grad()
        loss_fn(cnn(x.to(device)), y.to(device)).backward()
        opt.step()

preds, trues = [], []
cnn.eval()
with torch.no_grad():
    for x,y in test_loader:
        preds.extend(cnn(x.to(device)).cpu().numpy())
        trues.extend(y.numpy())

all_preds["CNN"] = np.array(preds)
all_true["CNN"]  = np.array(trues)

# =====================================================
# 5. LSTM
# =====================================================
print("\n🧠 Training LSTM")
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), 64)
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[0], h[1]), dim=1)
        return torch.sigmoid(self.fc(h)).squeeze()

lstm = LSTM().to(device)
opt = torch.optim.Adam(lstm.parameters(), lr=1e-3)

for epoch in range(2):
    print(f"  LSTM Epoch {epoch+1}/2")
    for x,y in train_loader:
        opt.zero_grad()
        loss_fn(lstm(x.to(device)), y.to(device)).backward()
        opt.step()

preds, trues = [], []
lstm.eval()
with torch.no_grad():
    for x,y in test_loader:
        preds.extend(lstm(x.to(device)).cpu().numpy())
        trues.extend(y.numpy())

all_preds["LSTM"] = np.array(preds)
all_true["LSTM"]  = np.array(trues)

# =====================================================
# 6. BERT (PROPOSED)
# =====================================================
print("\n🤖 Training BERT (Proposed)")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BertDS(Dataset):
    def __init__(self, texts, labels):
        self.t = texts.tolist()
        self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        enc = tokenizer(
            self.t[i], max_length=128,
            padding="max_length", truncation=True,
            return_tensors="pt"
        )
        return enc["input_ids"].squeeze(), enc["attention_mask"].squeeze(), torch.tensor(self.y[i], dtype=torch.float)

train_bert = BertDS(X_train, y_train)
test_bert  = BertDS(X_test, y_test)

train_loader = DataLoader(train_bert, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_bert, batch_size=8)

class BertReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768,1)
    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)
        return torch.sigmoid(self.fc(out.pooler_output)).squeeze()

bert = BertReg().to(device)
opt = torch.optim.AdamW(bert.parameters(), lr=2e-5)

for ids,mask,y in train_loader:
    opt.zero_grad()
    loss_fn(bert(ids.to(device), mask.to(device)), y.to(device)).backward()
    opt.step()

preds, trues = [], []
bert.eval()
with torch.no_grad():
    for ids,mask,y in test_loader:
        preds.extend(bert(ids.to(device), mask.to(device)).cpu().numpy())
        trues.extend(y.numpy())

all_preds["BERT (Proposed)"] = np.array(preds)
all_true["BERT (Proposed)"]  = np.array(trues)

# =====================================================
# 7. MAE PLOTS FOR 20%, 40%, 60%
# =====================================================
print("\n📉 Generating MAE plots (20%, 40%, 60%)")

low_bins = {
    "20%": (0.0, 0.2),
    "40%": (0.2, 0.4),
    "60%": (0.4, 0.6)
}

for band, (low, high) in low_bins.items():
    maes = {}
    for model in all_preds:
        y_true = all_true[model]
        y_pred = all_preds[model]
        idx = np.where((y_true >= low) & (y_true < high))[0]
        if len(idx) < 20:
            continue
        maes[model] = mean_absolute_error(y_true[idx], y_pred[idx])

    plt.figure(figsize=(10,5))
    plt.bar(maes.keys(), maes.values())
    plt.ylabel("MAE")
    plt.title(f"MAE at {band} Severity")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"mae_{band}_severity.png", dpi=300)
    plt.show()

# =====================================================
# 8. EMERGENCY RECALL PLOTS FOR 80%, 100%
# =====================================================
print("\n📈 Generating Emergency Recall plots (80%, 100%)")

high_bins = {
    "80%": (0.6, 0.8),
    "100%": (0.8, 1.0)
}

for band, (low, high) in high_bins.items():
    recalls = {}
    for model in all_preds:
        y_true = all_true[model]
        y_pred = all_preds[model]
        idx = np.where((y_true >= low) & (y_true < high))[0]
        if len(idx) < 10:
            continue
        recalls[model] = recall_score(
            to_emergency(y_true[idx]),
            to_emergency(y_pred[idx]),
            zero_division=0
        )

    plt.figure(figsize=(10,5))
    plt.bar(recalls.keys(), recalls.values())
    plt.ylim(0,1)
    plt.ylabel("Emergency Recall")
    plt.title(f"Emergency Recall at {band} Severity")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(f"recall_{band}_severity.png", dpi=300)
    plt.show()

# =====================================================
# 9. COMBINED SEVERITY-WISE LINE PLOT (FINAL FIGURE)
# =====================================================
print("\n📊 Generating combined severity-wise line plot")

severity_bins = {
    "20%": (0.0, 0.2),
    "40%": (0.2, 0.4),
    "60%": (0.4, 0.6),
    "80%": (0.6, 0.8),
    "100%": (0.8, 1.0)
}

plt.figure(figsize=(12,6))
for model in all_preds:
    values = []
    for band, (low, high) in severity_bins.items():
        y_true = all_true[model]
        y_pred = all_preds[model]
        idx = np.where((y_true >= low) & (y_true < high))[0]

        if band in ["80%", "100%"]:
            if len(idx) < 10:
                values.append(np.nan)
            else:
                values.append(
                    recall_score(
                        to_emergency(y_true[idx]),
                        to_emergency(y_pred[idx]),
                        zero_division=0
                    )
                )
        else:
            if len(idx) < 20:
                values.append(np.nan)
            else:
                values.append(
                    mean_absolute_error(y_true[idx], y_pred[idx])
                )

    plt.plot(severity_bins.keys(), values, marker="o", label=model)

plt.xlabel("Severity Level")
plt.ylabel("MAE (Low/Mid)  |  Recall (High)")
plt.title("Severity-Aware Model Performance Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("severity_aware_combined_plot.png", dpi=300)
plt.show()

print("\n✅ ALL GRAPHS GENERATED SUCCESSFULLY")
