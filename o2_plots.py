import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

y_true = np.load("o2_y_true.npy")
y_pred = np.load("o2_y_pred.npy")
y_prob = np.load("o2_y_prob.npy")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure()
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix (O2 Emergency)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.savefig("o2_confusion_matrix.png")

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (O2 Emergency)")
plt.legend()
plt.savefig("o2_roc_curve.png")

print("O2 graphs saved")
