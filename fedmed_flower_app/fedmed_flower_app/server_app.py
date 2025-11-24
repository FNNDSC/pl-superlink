"""FedMed Flower ServerApp."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .task import (
    DEFAULT_BATCH_SIZE,
    SimpleCNN,
    evaluate_model,
    get_test_loader,
    initial_parameters,
    load_model_parameters,
)

SERVER_SUMMARY_TOKEN = "[fedmed-superlink-app] SUMMARY "
CKPT_ENV_VAR = "FEDMED_OUTPUT_CKPT"

app = ServerApp()


def _central_evaluate(
    server_round: int,
    arrays: ArrayRecord,
    *,
    testloader,
    device: torch.device,
) -> MetricRecord:
    net = SimpleCNN()
    load_model_parameters(net, arrays)
    loss, accuracy = evaluate_model(net, testloader, device)
    return MetricRecord(
        {
            "server_round": server_round,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "num_examples": len(testloader.dataset),
        }
    )


def _metrics_to_dict(metrics: Dict[int, MetricRecord]) -> Dict[int, Dict[str, Any]]:
    return {round_id: dict(record) for round_id, record in metrics.items()}


def _write_checkpoint(arrays: ArrayRecord) -> str | None:
    ckpt_path = os.environ.get(CKPT_ENV_VAR)
    if not ckpt_path:
        return None
    try:
        path = Path(ckpt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        net = SimpleCNN()
        load_model_parameters(net, arrays)
        torch.save(net.state_dict(), path)
        return str(path)
    except Exception as exc:
        print(f"[fedmed-superlink-app] checkpoint-error: {exc}", flush=True)
        return None


def _emit_summary(result, context: Context) -> Dict[str, Any]:  # type: ignore[override]
    checkpoint_path = _write_checkpoint(result.arrays)
    summary = {
        "run_id": context.run_id,
        "rounds": context.run_config.get("num-server-rounds"),
        "total_clients": context.run_config.get("total-clients"),
        "train_metrics": _metrics_to_dict(result.train_metrics_clientapp),
        "evaluate_metrics": _metrics_to_dict(result.evaluate_metrics_clientapp),
        "server_metrics": _metrics_to_dict(result.evaluate_metrics_serverapp),
    }
    if checkpoint_path:
        summary["checkpoint"] = checkpoint_path
    print(f"{SERVER_SUMMARY_TOKEN}{json.dumps(summary)}", flush=True)
    return summary


@app.main()
def main(grid: Grid, context: Context) -> None:
    """FedMed ServerApp entrypoint."""

    run_config = context.run_config
    total_clients = int(run_config.get("total-clients", 1))
    num_rounds = int(run_config.get("num-server-rounds", 1))
    local_epochs = int(run_config.get("local-epochs", 2))
    learning_rate = float(run_config.get("learning-rate", 1e-3))
    data_seed = int(run_config.get("data-seed", 13))
    batch_size = int(run_config.get("batch-size", DEFAULT_BATCH_SIZE))
    fraction_train = float(run_config.get("fraction-fit", 1.0))
    fraction_evaluate = float(run_config.get("fraction-evaluate", 1.0))
    test_batch_size = int(run_config.get("test-batch-size", max(128, batch_size)))

    strategy = FedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=max(1, int(total_clients * fraction_train)),
        min_evaluate_nodes=max(1, int(total_clients * fraction_evaluate)),
        min_available_nodes=total_clients,
    )

    train_config = ConfigRecord(
        {
            "batch-size": batch_size,
            "local-epochs": local_epochs,
            "learning-rate": learning_rate,
            "total-clients": total_clients,
            "data-seed": data_seed,
        }
    )

    testloader = get_test_loader(batch_size=test_batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_parameters(),
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=lambda server_round, arrays: _central_evaluate(
            server_round,
            arrays,
            testloader=testloader,
            device=device,
        ),
    )

    _emit_summary(result, context)
