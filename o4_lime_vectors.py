import torch
import numpy as np
import pandas as pd
import shap

from lime.lime_text import LimeTextExplainer
from transformers import AutoTokenizer
from o2_model import AsymCostFed

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 32
ALPHA = 0.5   # fusion weight: LIME vs DeepSHAP

print("Using device:", DEVICE)

# =====================================================
# LOAD MODEL
# =====================================================
model = AsymCostFed().to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# =====================================================
# LIME prediction function
# =====================================================
def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]

    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    with torch.no_grad():
        probs = model(
            tokens["input_ids"],
            tokens["attention_mask"]
        )

    probs = probs.cpu().numpy().reshape(-1, 1)
    return np.hstack([1 - probs, probs])

# =====================================================
# LOAD CLIENT DATA
# =====================================================
clients = []
GLOBAL_VOCAB = set()

for i in range(3):
    df = pd.read_csv(f"client_{i}.csv")
    text = str(df.iloc[0]["symptom_text"])
    clients.append(text)
    GLOBAL_VOCAB.update(text.lower().split())

GLOBAL_VOCAB = sorted(GLOBAL_VOCAB)
vocab_index = {w: i for i, w in enumerate(GLOBAL_VOCAB)}

print("📚 Global vocab size:", len(GLOBAL_VOCAB))

# =====================================================
# LIME SETUP
# =====================================================
lime_explainer = LimeTextExplainer(
    class_names=["No-Emergency", "Emergency"]
)

# =====================================================
# DeepSHAP SETUP (EMBEDDING LEVEL)
# =====================================================
bg_tokens = tokenizer(
    clients,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN
).to(DEVICE)

bg_embeddings = model.encoder.embeddings.word_embeddings(
    bg_tokens["input_ids"]
)
bg_attention = bg_tokens["attention_mask"].float()

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeddings, attention_mask):
        return self.model.forward_from_embeddings(
            embeddings, attention_mask
        )

wrapped_model = WrappedModel(model)

deep_explainer = shap.DeepExplainer(
    wrapped_model,
    [bg_embeddings, bg_attention]
)

# =====================================================
# GENERATE EXPLANATION VECTORS
# =====================================================
vectors = []

for idx, text in enumerate(clients):
    print(f"\n🔍 Explaining client {idx}")

    # ---------- LIME ----------
    lime_exp = lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=15
    )

    lime_vec = np.zeros(len(GLOBAL_VOCAB))
    for w, s in lime_exp.as_list():
        w = w.lower()
        if w in vocab_index:
            lime_vec[vocab_index[w]] = s

    # ---------- DeepSHAP ----------
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    ).to(DEVICE)

    embeddings = model.encoder.embeddings.word_embeddings(
        tokens["input_ids"]
    )
    attention = tokens["attention_mask"].float()

    # 🔴 IMPORTANT FIX: disable additivity check
    shap_vals = deep_explainer.shap_values(
        [embeddings, attention],
        check_additivity=False
    )

    shap_scores = shap_vals[0][0].sum(axis=1).reshape(-1)
    words = tokenizer.convert_ids_to_tokens(
        tokens["input_ids"][0].cpu().numpy()
    )

    shap_vec = np.zeros(len(GLOBAL_VOCAB))
    for w, v in zip(words, shap_scores):
        w = w.replace("##", "").lower()
        if w in vocab_index:
            shap_vec[vocab_index[w]] += float(v)

    # ---------- FUSION ----------
    fused_vec = ALPHA * lime_vec + (1 - ALPHA) * shap_vec
    vectors.append(fused_vec)

# =====================================================
# SAVE VECTORS
# =====================================================
vectors = np.array(vectors)
np.save("o4_lime_deepshap_vectors.npy", vectors)

print("\n✅ LIME + DeepSHAP fused vectors saved")
print("Shape:", vectors.shape)
