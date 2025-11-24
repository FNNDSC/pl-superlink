"""FedMed Flower ClientApp."""

from __future__ import annotations

import json
from typing import Any, Dict

import torch
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .task import (
    DEFAULT_BATCH_SIZE,
    SimpleCNN,
    get_model_parameters,
    load_data,
    load_model_parameters,
    train_model,
    evaluate_model,
)

CLIENT_SUMMARY_TOKEN = "[fedmed-supernode-app] SUMMARY "

app = ClientApp()


def _get_node_config(context: Context) -> Dict[str, Any]:
    """Resolve node configuration with sensible defaults."""
    run_config = context.run_config
    node_config = context.node_config or {}
    return {
        "partition-id": int(node_config.get("partition-id", 0)),
        "num-partitions": int(
            node_config.get("num-partitions", run_config.get("total-clients", 1))
        ),
        "data-seed": int(node_config.get("data-seed", run_config.get("data-seed", 13))),
    }


def _log_metrics(kind: str, partition_id: int, metrics: Dict[str, Any]) -> None:
    payload = {"kind": kind, "partition_id": partition_id, "metrics": metrics}
    print(f"{CLIENT_SUMMARY_TOKEN}{json.dumps(payload)}", flush=True)


def _resolve_train_hyperparams(config: Dict[str, Any], run_config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "batch-size": int(config.get("batch-size", run_config.get("batch-size", DEFAULT_BATCH_SIZE))),
        "local-epochs": int(config.get("local-epochs", run_config.get("local-epochs", 2))),
        "learning-rate": float(config.get("learning-rate", run_config.get("learning-rate", 1e-3))),
    }


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Handle train instructions coming from the ServerApp."""
    node_config = _get_node_config(context)
    partition_id = node_config["partition-id"]
    num_partitions = node_config["num-partitions"]
    data_seed = node_config["data-seed"]

    config = msg.content.get("config", {})
    hyper = _resolve_train_hyperparams(config, context.run_config)
    batch_size = hyper["batch-size"]
    local_epochs = hyper["local-epochs"]
    learning_rate = hyper["learning-rate"]

    trainloader, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size=batch_size,
        seed=data_seed,
    )

    net = SimpleCNN()
    load_model_parameters(net, msg.content["arrays"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loss = train_model(
        net,
        trainloader,
        epochs=local_epochs,
        device=device,
        lr=learning_rate,
    )
    val_loss, val_accuracy = evaluate_model(net, valloader, device)

    metrics = {
        "num-examples": len(trainloader.dataset),
        "loss": float(val_loss),
        "accuracy": float(val_accuracy),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "val_accuracy": float(val_accuracy),
    }
    _log_metrics("train", partition_id, metrics)

    reply = RecordDict(
        {
            "arrays": get_model_parameters(net),
            "metrics": MetricRecord(metrics),
        }
    )
    return Message(content=reply, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Handle evaluate instructions coming from the ServerApp."""
    node_config = _get_node_config(context)
    partition_id = node_config["partition-id"]
    num_partitions = node_config["num-partitions"]
    data_seed = node_config["data-seed"]

    batch_size = int(context.run_config.get("batch-size", DEFAULT_BATCH_SIZE))
    _, valloader = load_data(
        partition_id,
        num_partitions,
        batch_size=batch_size,
        seed=data_seed,
    )

    net = SimpleCNN()
    load_model_parameters(net, msg.content["arrays"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loss, accuracy = evaluate_model(net, valloader, device)
    metrics = {
        "num-examples": len(valloader.dataset),
        "loss": float(loss),
        "accuracy": float(accuracy),
    }
    _log_metrics("evaluate", partition_id, metrics)

    reply = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=reply, reply_to=msg)
