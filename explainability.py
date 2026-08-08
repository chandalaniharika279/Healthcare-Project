from lime.lime_text import LimeTextExplainer
import numpy as np

class EmergencyWrapper:
    def __init__(self, df):
        self.df = df

    def predict_proba(self, texts):
        probs = []
        for t in texts:
            match = self.df[self.df["symptom_text"].str.contains(t, case=False, na=False)]
            if match.empty:
                probs.append([0.95, 0.05])
            else:
                p = match["emergency"].mean()
                probs.append([1-p, p])
        return np.array(probs)

def lime_explain(text, df):
    explainer = LimeTextExplainer(class_names=["No Emergency", "Emergency"])
    model = EmergencyWrapper(df)
    exp = explainer.explain_instance(text, model.predict_proba, num_features=6)
    return exp.as_list()