import torch
import torch.nn as nn
from transformers import AutoModel

RED_FLAGS = [
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "loss of consciousness",
    "severe bleeding"
]

class AsymCostFed(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    # ===============================
    # NORMAL FORWARD (USED IN TRAINING)
    # ===============================
    def forward(self, input_ids, attention_mask, text=None):
        if text:
            t = text.lower()
            for rf in RED_FLAGS:
                if rf in t:
                    # rule-based emergency override
                    return torch.tensor([[0.95]], device=input_ids.device)

        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        return self.classifier(cls)

    # ===============================
    # REQUIRED FOR DeepSHAP
    # ===============================
    def forward_from_embeddings(self, embeddings, attention_mask):
        """
        embeddings: output of BERT word embeddings
        """
        cls = self.encoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0]

        return self.classifier(cls)
