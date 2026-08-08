import flwr as fl
import numpy as np

# =====================================================
# Metric aggregation (weighted global loss)
# =====================================================
def weighted_average(metrics):
    losses = []
    examples = []

    for num_examples, metric in metrics:
        losses.append(metric["loss"])
        examples.append(num_examples)

    return {
        "loss": np.average(losses, weights=examples)
    }

# =====================================================
# FedAvg strategy
# =====================================================
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# =====================================================
# Start server and save history
# =====================================================
if __name__ == "__main__":
    print("🚀 Starting Federated Learning Server (O5)")

    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # -------------------------------------------------
    # Save federated loss automatically
    # -------------------------------------------------
    fed_loss = [loss for _, loss in history.losses_distributed]
    np.save("o5_federated_loss.npy", np.array(fed_loss))

    print("✅ Federated loss saved to o5_federated_loss.npy")
