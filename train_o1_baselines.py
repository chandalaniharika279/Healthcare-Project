import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import os

# ---------------- LOAD DATA ----------------
df = pd.read_csv("D:/Nandini/data/hospital_triage.csv")
df.columns = df.columns.str.lower().str.strip()

# ---------------- DETECT TEXT COLUMN ----------------
TEXT_CANDIDATES = [
    "symptom_text",
    "chiefcomplaint",
    "cc_coldlikesymptoms",
    "cc_abdominalcramping",
]

text_col = None
for c in TEXT_CANDIDATES:
    if c in df.columns:
        text_col = c
        break

if text_col is None:
    raise ValueError("❌ No valid symptom text column found")

print(f"✅ Using text column: {text_col}")

# ---------------- TARGET: SEVERITY ----------------
if "severity_score" in df.columns:
    df["severity"] = pd.to_numeric(df["severity_score"], errors="coerce")
else:
    print("⚠️ severity_score not found → deriving from ESI")
    # ESI: 1 (critical) → 1.0, 5 (non-urgent) → 0.2
    df["severity"] = (6 - df["esi"]) / 5

# ---------------- CLEAN DATA ----------------
df = df[[text_col, "severity"]].dropna()
df = df[df[text_col].astype(str).str.strip() != ""]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"🧹 Clean rows → Train: {len(train_df)}, Test: {len(test_df)}")

# ---------------- TF-IDF (FIXED) ----------------
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    lowercase=True
)

X_train = vectorizer.fit_transform(train_df[text_col].astype(str))
X_test = vectorizer.transform(test_df[text_col].astype(str))

y_train = train_df["severity"].values
y_test = test_df["severity"].values

print("📐 Vocabulary size:", len(vectorizer.vocabulary_))

# ---------------- MODELS (REGRESSION ONLY) ----------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
}

os.makedirs("baseline_preds", exist_ok=True)

# ---------------- TRAIN & SAVE ----------------
print("\n🔁 Training baseline models\n")

results = []

for name, model in models.items():
    print(f"▶ {name}")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print(f"   MAE: {mae:.4f}")

    # Save for bucket-wise analysis
    np.save(f"baseline_preds/{name}_pred.npy", preds)

    results.append((name, mae))

# Save ground truth once
np.save("baseline_preds/y_true.npy", y_test)

print("\n✅ Baseline training completed successfully")

# ---------------- SUMMARY ----------------
print("\n📊 Baseline MAE Summary")
for name, mae in results:
    print(f"{name:20s} → MAE = {mae:.4f}")
