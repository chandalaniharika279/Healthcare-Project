import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# Load federated loss (AUTO-GENERATED)
# =====================================================
fed_loss = np.load("o5_federated_loss.npy")
rounds = np.arange(1, len(fed_loss) + 1)

# =====================================================
# Plot
# =====================================================
plt.figure()
plt.plot(rounds, fed_loss, marker="o")
plt.xlabel("Federated Rounds")
plt.ylabel("Global MSE Loss")
plt.title("O5: Federated Severity Regression Convergence")
plt.grid(True)

plt.savefig("o5_federated_convergence.png", dpi=300)
plt.show()

print("✅ Plot saved as o5_federated_convergence.png")
