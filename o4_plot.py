import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Load similarity matrix
# =====================================================
sim = np.load("o4_similarity_matrix.npy")

num_clients = sim.shape[0]
client_labels = [f"Client {i+1}" for i in range(num_clients)]

# =====================================================
# Plot heatmap
# =====================================================
plt.figure(figsize=(6, 5))
im = plt.imshow(sim, cmap="viridis", vmin=0, vmax=1)

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label("Cosine Similarity", fontsize=11)

# Axis labels
plt.xticks(range(num_clients), client_labels, rotation=45)
plt.yticks(range(num_clients), client_labels)

plt.xlabel("Clients")
plt.ylabel("Clients")
plt.title("LIME + Deep SHAP Explanation Consistency Across Clients")

# =====================================================
# Annotate similarity values
# =====================================================
for i in range(num_clients):
    for j in range(num_clients):
        plt.text(
            j, i,
            f"{sim[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if sim[i, j] < 0.5 else "black",
            fontsize=9
        )

plt.tight_layout()
plt.savefig("o4_explanation_consistency.png", dpi=300)
plt.show()

print("✅ O4 explanation consistency plot saved as o4_explanation_consistency.png")
