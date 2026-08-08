# =====================================================
# FAST + TRACEABLE MODEL COMPARISON (CNN + LSTM + BERT)
# OFFLINE-SAFE + GPU-CORRECT VERSION
# =====================================================

import os
import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error

from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ================= OFFLINE MODE ======================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

BERT_PATH = "D:/models/bert-base-uncased"

# ================= TRACE HELPER ======================
def trace(msg):
    print(f"\n[TRACE] {msg}")

# =====================================================
# 0. SEED + DEVICE
# =====================================================
trace("Setting seeds and device")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🚀 Device:", device)

torch.backends.cudnn.benchmark = True

# =====================================================
# 1. LOAD DATA (FULL DATASET)
# =====================================================
trace("Loading dataset")

train_df = pd.read_csv("D:/Nandini/tra_in_o1.csv")
test_df  = pd.read_csv("D:/Nandini/te_st_o1.csv")

TEXT_COL = "symptom_text"
TARGET = "severity_score"

train_df = train_df.dropna(subset=[TEXT_COL, TARGET])
test_df  = test_df.dropna(subset=[TEXT_COL, TARGET])

X_train = train_df[TEXT_COL].astype(str)
X_test  = test_df[TEXT_COL].astype(str)
y_train = train_df[TARGET].values
y_test  = test_df[TARGET].values

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

results = {}

# =====================================================
# 2. TF-IDF BASELINES (CPU)
# =====================================================
trace("Running TF-IDF baseline models")

vectorizer = TfidfVectorizer(
    max_features=3000,
    min_df=20,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

lin = LinearRegression(n_jobs=-1)
lin.fit(X_train_tfidf, y_train)
results["Linear Regression"] = mean_absolute_error(
    y_test, lin.predict(X_test_tfidf)
)

y_train_cls = np.clip((y_train * 5).round().astype(int), 1, 5)
log = LogisticRegression(max_iter=500, n_jobs=-1)
log.fit(X_train_tfidf, y_train_cls)
results["Logistic Regression"] = mean_absolute_error(
    y_test, log.predict(X_test_tfidf) / 5.0
)

rf = RandomForestRegressor(
    n_estimators=30,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_tfidf, y_train)
results["Random Forest"] = mean_absolute_error(
    y_test, rf.predict(X_test_tfidf)
)

svm = LinearSVR()
svm.fit(X_train_tfidf, y_train)
results["SVM"] = mean_absolute_error(
    y_test, svm.predict(X_test_tfidf)
)

# =====================================================
# 3. SEQUENCE DATA (CNN / LSTM)
# =====================================================
trace("Preparing sequence data")

MAX_VOCAB = 8000
MAX_LEN = 40

counter = Counter(" ".join(X_train).split())
vocab = {"<pad>": 0, "<unk>": 1}
for w, _ in counter.most_common(MAX_VOCAB - 2):
    vocab[w] = len(vocab)

def encode(text):
    ids = [vocab.get(t, 1) for t in text.split()][:MAX_LEN]
    return ids + [0] * (MAX_LEN - len(ids))

class SeqDataset(Dataset):
    def __init__(self, texts, labels):
        self.X = [encode(t) for t in texts]
        self.y = labels

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.y)

train_loader = DataLoader(
    SeqDataset(X_train, y_train),
    batch_size=64,
    shuffle=True,
    pin_memory=True
)

test_loader = DataLoader(
    SeqDataset(X_test, y_test),
    batch_size=64,
    pin_memory=True
)

loss_fn = nn.MSELoss()

# =====================================================
# 4. CNN (GPU)
# =====================================================
trace("Training CNN on GPU")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), 64)
        self.conv = nn.Conv1d(64, 64, 3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.emb(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = torch.max(x, 2)[0]
        return torch.sigmoid(self.fc(x)).squeeze()

cnn = CNN().to(device)
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)

for epoch in range(2):
    trace(f"CNN Epoch {epoch+1}")
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(cnn(x), y).backward()
        opt.step()

cnn.eval()
preds, trues = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds.extend(cnn(x.to(device)).cpu().numpy())
        trues.extend(y.numpy())

results["CNN"] = mean_absolute_error(trues, preds)

# =====================================================
# 5. LSTM (GPU)
# =====================================================
trace("Training LSTM on GPU")

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
    trace(f"LSTM Epoch {epoch+1}")
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss_fn(lstm(x), y).backward()
        opt.step()

lstm.eval()
preds, trues = [], []
with torch.no_grad():
    for x, y in test_loader:
        preds.extend(lstm(x.to(device)).cpu().numpy())
        trues.extend(y.numpy())

results["LSTM"] = mean_absolute_error(trues, preds)

# =====================================================
# 6. BERT (OFFLINE SAFE)
# =====================================================
if os.path.exists(BERT_PATH):
    trace("Training BERT from local files")

    tokenizer = BertTokenizer.from_pretrained(
        BERT_PATH, local_files_only=True
    )

    class BertDS(Dataset):
        def __init__(self, texts, labels):
            self.t = texts.tolist()
            self.y = labels

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            enc = tokenizer(
                self.t[i],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return (
                enc["input_ids"].squeeze(),
                enc["attention_mask"].squeeze(),
                torch.tensor(self.y[i], dtype=torch.float32)
            )

    class BertReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel.from_pretrained(
                BERT_PATH, local_files_only=True
            )
            self.fc = nn.Linear(768, 1)

        def forward(self, ids, mask):
            out = self.bert(ids, attention_mask=mask)
            return torch.sigmoid(self.fc(out.pooler_output)).squeeze()

    bert = BertReg().to(device)
    opt = torch.optim.AdamW(bert.parameters(), lr=2e-5)

    train_loader = DataLoader(BertDS(X_train, y_train), batch_size=8, shuffle=True)
    test_loader  = DataLoader(BertDS(X_test, y_test), batch_size=8)

    for epoch in range(2):
        trace(f"BERT Epoch {epoch+1}")
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(bert(ids, mask), y).backward()
            opt.step()

    bert.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ids, mask, y in test_loader:
            preds.extend(bert(ids.to(device), mask.to(device)).cpu().numpy())
            trues.extend(y.numpy())

    results["BERT (Proposed)"] = mean_absolute_error(trues, preds)

else:
    trace("BERT files not found → skipping BERT")

# =====================================================
# 7. FINAL GRAPH
# =====================================================
trace("Plotting final MAE comparison")

plt.figure(figsize=(10,6))
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45, ha="right")
plt.ylabel("Mean Absolute Error (↓ better)")
plt.title("Model Comparison on Full Dataset")
plt.grid(axis="y")
plt.tight_layout()
plt.show()

trace("EXECUTION COMPLETED SUCCESSFULLY")
