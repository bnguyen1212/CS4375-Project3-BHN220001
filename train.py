"""Training utilities for logits-based PyTorch classification models.

This module is model-agnostic and works with both MLP and CNN architectures
as long as the model returns raw logits compatible with CrossEntropyLoss.
"""

from __future__ import annotations

from copy import deepcopy
from time import perf_counter
from typing import Any, Mapping

import torch
from torch import nn
from torch.optim import Adam, SGD, Optimizer
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate average loss and accuracy on a dataloader.

    Args:
        model: PyTorch model that outputs raw logits.
        loader: DataLoader for validation or test data.
        criterion: Loss function (typically CrossEntropyLoss).
        device: torch.device("cpu") or torch.device("cuda").

    Returns:
        avg_loss: Mean loss across all samples.
        accuracy: Fraction of correct predictions in [0, 1].
    """

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def _build_optimizer(model: nn.Module, config: Mapping[str, Any]) -> Optimizer:
    """Create optimizer from configuration."""

    required_keys = ["lr", "optimizer", "epochs"]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    lr = float(config["lr"])
    optimizer_name = str(config["optimizer"]).lower()
    weight_decay = float(config.get("weight_decay", 0.0))

    if lr <= 0:
        raise ValueError("config['lr'] must be > 0")

    if optimizer_name == "sgd":
        return SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError("config['optimizer'] must be either 'sgd' or 'adam'")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]], float]:
    """Train a model with validation monitoring and optional early stopping.

    Expected config keys:
        lr: Learning rate (required)
        optimizer: "sgd" or "adam" (required)
        epochs: Max training epochs (required)
        weight_decay: L2 penalty (optional, default 0.0)
        patience: Early stopping patience in epochs (optional)

    Returns:
        best_model: Model restored to best validation-accuracy checkpoint.
        history: Dict with lists for train_loss, val_loss, val_acc.
        runtime_seconds: Total wall-clock training time.
    """

    epochs = int(config["epochs"])
    if epochs <= 0:
        raise ValueError("config['epochs'] must be > 0")

    patience = config.get("patience", None)
    if patience is not None:
        patience = int(patience)
        if patience <= 0:
            raise ValueError("config['patience'] must be > 0 when provided")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _build_optimizer(model, config)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_state = deepcopy(model.state_dict())
    epochs_without_improvement = 0

    start_time = perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = running_loss / total_samples if total_samples > 0 else 0.0
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            print(
                "Early stopping triggered: "
                f"no validation accuracy improvement for {patience} epoch(s)."
            )
            break

    runtime_seconds = perf_counter() - start_time

    model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    return model, history, runtime_seconds
