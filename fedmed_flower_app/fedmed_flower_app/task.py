"""PneumoniaMNIST model and dataset utilities used by the FedMed Flower app."""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.app import Array, ArrayRecord
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

DATA_FLAG = "pneumoniamnist"
DEFAULT_BATCH_SIZE = 64
_DATA_CACHE: Dict[str, Dataset] = {}


class SimpleCNN(nn.Module):
    """Simple CNN tuned for PneumoniaMNIST (grayscale, binary classification)."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def _data_root() -> Path:
    """Default root used by MedMNIST downloads."""
    root_env = os.environ.get("MEDMNIST_ROOT")
    if root_env:
        root = Path(root_env).expanduser()
    else:
        flwr_home = os.environ.get("FLWR_HOME")
        if flwr_home:
            root = Path(flwr_home).expanduser() / "datasets" / "medmnist"
        else:
            root = Path.home() / ".cache" / "medmnist"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_dataset(split: str) -> Dataset:
    cache_key = f"{DATA_FLAG}:{split}"
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    info = INFO[DATA_FLAG]
    data_class = getattr(medmnist, info["python_class"])
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = data_class(
        split=split,
        download=True,
        transform=transform,
        size=28,
        root=str(_data_root()),
    )
    _DATA_CACHE[cache_key] = dataset
    return dataset


def _partition_indices(
    num_samples: int,
    num_partitions: int,
    partition_id: int,
    seed: int,
) -> np.ndarray:
    if num_partitions <= 0:
        raise ValueError("num_partitions must be positive")
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError("partition_id out of range")
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples)
    rng.shuffle(indices)
    partitions = np.array_split(indices, num_partitions)
    return partitions[partition_id]


def load_data(
    partition_id: int,
    num_partitions: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 13,
) -> Tuple[DataLoader, DataLoader]:
    """Return (trainloader, valloader) for the requested logical client."""
    train_dataset = _load_dataset("train")
    val_dataset = _load_dataset("val")
    indices = _partition_indices(len(train_dataset), num_partitions, partition_id, seed)
    train_subset = Subset(train_dataset, indices.tolist())
    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return trainloader, valloader


def get_test_loader(batch_size: int = 128) -> DataLoader:
    """Return the held-out test split."""
    test_dataset = _load_dataset("test")
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def _dataset_labels(dataset: Dataset) -> torch.Tensor:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        labels = np.asarray(base.labels)[dataset.indices]
    else:
        labels = np.asarray(dataset.labels)
    labels = torch.as_tensor(labels).view(-1).long()
    return labels


def _class_weights(dataset: Dataset) -> torch.Tensor:
    labels = _dataset_labels(dataset)
    counts = torch.bincount(labels, minlength=2).float()
    weights = counts.sum() / (counts + 1e-9)
    return weights


def train_model(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> float:
    """Run local training and return the average training loss."""
    net.to(device)
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(trainloader.dataset).to(device)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    total_loss = 0.0
    total_batches = 0
    for _ in range(epochs):
        for data, target in trainloader:
            data = data.to(device)
            target = target.squeeze().long().to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches if total_batches else 0.0


def evaluate_model(
    net: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (loss, accuracy) for the provided DataLoader."""
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.squeeze().long().to(device)
            outputs = net(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / len(loader) if len(loader) else 0.0
    accuracy = correct / total if total else 0.0
    return avg_loss, accuracy


def _base_state_dict() -> OrderedDict[str, torch.Tensor]:
    """Return a freshly initialized model state dict."""
    return SimpleCNN().state_dict()


PARAMETER_NAMES = tuple(_base_state_dict().keys())


def _state_dict_to_numpy(state_dict: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, np.ndarray]:
    return OrderedDict(
        (name, tensor.detach().cpu().numpy().astype(np.float32))
        for name, tensor in state_dict.items()
    )


def _numpy_to_state_dict(
    arrays: OrderedDict[str, np.ndarray],
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, torch.tensor(arrays[name]))
        for name in PARAMETER_NAMES
    )


def encode_parameters(state_dict_np: OrderedDict[str, np.ndarray]) -> ArrayRecord:
    """Convert an OrderedDict of numpy arrays into an ArrayRecord."""
    record = ArrayRecord()
    for name in PARAMETER_NAMES:
        record[name] = Array.from_numpy_ndarray(state_dict_np[name].astype(np.float32))
    return record


def decode_parameters(record: ArrayRecord) -> OrderedDict[str, np.ndarray]:
    """Convert an ArrayRecord back into numpy arrays keyed by param name."""
    return OrderedDict(
        (name, record[name].numpy().astype(np.float32)) for name in PARAMETER_NAMES
    )


def load_model_parameters(net: nn.Module, arrays: ArrayRecord) -> None:
    """Load Flower parameters into the provided network."""
    state_dict = _numpy_to_state_dict(decode_parameters(arrays))
    net.load_state_dict(state_dict, strict=True)


def get_model_parameters(net: nn.Module) -> ArrayRecord:
    """Extract Flower-ready parameters from the provided network."""
    state_dict_np = _state_dict_to_numpy(net.state_dict())
    return encode_parameters(state_dict_np)


def initial_parameters() -> ArrayRecord:
    """Return the initial model parameters as an ArrayRecord."""
    return get_model_parameters(SimpleCNN())
