# =====================================================
# AUTOMATIC SEVERITY-WISE BEST MODEL ANALYSIS (FULL DATA)
# =====================================================

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
from sklearn.metrics import mean_absolute_error, recall_score

from transformers import BertTokenizer, BertModel
from torch import nn
from torch.utils.data import Dataset, DataLoader

# =====================================================
# 0. SETUP
# =====================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Device:", device)

EMERGENCY_THRESHOLD = 0.7

def to_emergency(y):
    return (y >= EMERGENCY_THRESHOLD).astype(int)

# =====================================================
# 1. LOAD FULL DATASET
# =====================================================
print("\n📥 Loading full dataset...")
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

all_preds = {}
all_true  = {}

# =====================================================
# 2. TF-IDF BASELINES (AUTO)
# =====================================================
print("\n🔤 Training TF-IDF baselines...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=10,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# Linear Regression
lin = LinearRegression(n_jobs=-1)
lin.fit(X_train_tfidf, y_train)
pred = lin.predict(X_test_tfidf)
all_preds["Linear Regression"] = pred
all_true["Linear Regression"]  = y_test

# Logistic Regression
y_train_cls = np.clip((y_train * 5).round().astype(int), 1, 5)
log = LogisticRegression(max_iter=500, n_jobs=-1)
log.fit(X_train_tfidf, y_train_cls)
pred = log.predict(X_test_tfidf) / 5.0
all_preds["Logistic Regression"] = pred
all_true["Logistic Regression"]  = y_test

# Random Forest
rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    n_jobs=-1,
    random_state=SEED
)
rf.fit(X_train_tfidf, y_train)
pred = rf.predict(X_test_tfidf)
all_preds["Random Forest"] = pred
all_true["Random Forest"]  = y_test

# SVM
svm = LinearSVR()
svm.fit(X_train_tfidf, y_train)
pred = svm.predict(X_test_tfidf)
all_preds["SVM"] = pred
all_true["SVM"]  = y_test

# =====================================================
# 3. SEQUENCE MODELS (CNN / LSTM)
# =====================================================
print("\n📚 Preparing sequence data...")
MAX_VOCAB = 10000
MAX_LEN = 40

counter = Counter(" ".join(X_train).split())
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

train_loader = DataLoader(train_seq, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_seq, batch_size=128)

# ---------- CNN ----------
print("\n🧠 Training CNN...")
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

for x,y in train_loader:
    opt.zero_grad()
    loss_fn(cnn(x.to(device)), y.to(device)).backward()
    opt.step()

cnn.eval()
preds = []
with torch.no_grad():
    for x,_ in test_loader:
        preds.extend(cnn(x.to(device)).cpu().numpy())

all_preds["CNN"] = np.array(preds)
all_true["CNN"]  = y_test

# ---------- LSTM ----------
print("\n🧠 Training LSTM...")
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

for x,y in train_loader:
    opt.zero_grad()
    loss_fn(lstm(x.to(device)), y.to(device)).backward()
    opt.step()

lstm.eval()
preds = []
with torch.no_grad():
    for x,_ in test_loader:
        preds.extend(lstm(x.to(device)).cpu().numpy())

all_preds["LSTM"] = np.array(preds)
all_true["LSTM"]  = y_test

# =====================================================
# 4. BERT (PROPOSED – SAFE FULL DATA)
# =====================================================
print("\n🤖 Training BERT (Proposed, frozen encoder)...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BertDS(Dataset):
    def __init__(self, texts, labels):
        self.t = texts.tolist()
        self.y = labels
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        enc = tokenizer(
            self.t[i],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return enc["input_ids"].squeeze(), enc["attention_mask"].squeeze(), torch.tensor(self.y[i], dtype=torch.float)

train_bert = BertDS(X_train, y_train)
test_bert  = BertDS(X_test, y_test)

train_loader = DataLoader(train_bert, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_bert, batch_size=16)

class BertReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for p in self.bert.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(768,1)
    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)
        return torch.sigmoid(self.fc(out.pooler_output)).squeeze()

bert = BertReg().to(device)
opt = torch.optim.AdamW(bert.fc.parameters(), lr=2e-4)

for ids,mask,y in train_loader:
    opt.zero_grad()
    loss_fn(bert(ids.to(device), mask.to(device)), y.to(device)).backward()
    opt.step()

bert.eval()
preds = []
with torch.no_grad():
    for ids,mask,_ in test_loader:
        preds.extend(bert(ids.to(device), mask.to(device)).cpu().numpy())

all_preds["BERT (Proposed)"] = np.array(preds)
all_true["BERT (Proposed)"]  = y_test

# =====================================================
# 5. SEVERITY-WISE BEST MODEL (AUTOMATIC)
# =====================================================
# =====================================================
# 5. AUTOMATIC SEVERITY-WISE BEST MODEL + PLOTS
# =====================================================
print("\n📊 Computing severity-wise best model (AUTO)...")

# Automatically derive severity bins from data (quantiles)
quantiles = np.quantile(y_test, [0, 0.2, 0.4, 0.6, 0.8, 1.0])

severity_bins = {
    f"{int(q*100)}%": (quantiles[i], quantiles[i+1])
    for i, q in enumerate([0.2, 0.4, 0.6, 0.8, 1.0])
}

records = []

for band, (low, high) in severity_bins.items():
    for model in all_preds:
        y_t = all_true[model]
        y_p = all_preds[model]

        idx = np.where((y_t >= low) & (y_t < high))[0]
        if len(idx) < 50:
            continue

        if high <= EMERGENCY_THRESHOLD:
            score = mean_absolute_error(y_t[idx], y_p[idx])
            metric = "MAE"
        else:
            score = recall_score(
                to_emergency(y_t[idx]),
                to_emergency(y_p[idx]),
                zero_division=0
            )
            metric = "Recall"

        records.append([band, model, score, metric])

results_df = pd.DataFrame(
    records,
    columns=["Severity Band", "Model", "Score", "Metric"]
)

# =====================================================
# 6. BEST MODEL PER SEVERITY BAND
# =====================================================
best_models = []

for band in results_df["Severity Band"].unique():
    subset = results_df[results_df["Severity Band"] == band]
    metric = subset["Metric"].iloc[0]

    if metric == "MAE":
        best_row = subset.loc[subset["Score"].idxmin()]
    else:
        best_row = subset.loc[subset["Score"].idxmax()]

    best_models.append(best_row)

best_df = pd.DataFrame(best_models)

print("\n✅ BEST MODEL PER SEVERITY LEVEL\n")
print(best_df)

best_df.to_csv("best_model_per_severity_full.csv", index=False)
# =====================================================
# 7. PLOTS
# =====================================================

# ---- Plot 1: Best model per severity ----
plt.figure(figsize=(12,6))
sns.barplot(
    data=best_df,
    x="Severity Band",
    y="Score",
    hue="Model"
)
plt.title("Best Model per Severity Level (Automatic)")
plt.ylabel("Score (MAE / Recall)")
plt.tight_layout()
plt.savefig("best_model_per_severity.png", dpi=300)
plt.show()

# ---- Plot 2: Model performance across severity ----
plt.figure(figsize=(14,6))
sns.lineplot(
    data=results_df,
    x="Severity Band",
    y="Score",
    hue="Model",
    marker="o"
)
plt.title("Model Performance Across Severity Levels")
plt.ylabel("MAE (Low/Mid) | Recall (High)")
plt.tight_layout()
plt.savefig("model_performance_across_severity.png", dpi=300)
plt.show()

# ---- Plot 3: Emergency Recall Only ----
high_sev = results_df[results_df["Metric"] == "Recall"]

plt.figure(figsize=(12,6))
sns.barplot(
    data=high_sev,
    x="Model",
    y="Score"
)
plt.title("Emergency Recall Comparison (High Severity Only)")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("emergency_recall_comparison.png", dpi=300)
plt.show()
