"""Configurable MLP model for image classification.

Design choices:
- Accepts image tensors and flattens them inside forward().
- Supports variable depth using a hidden_dims list.
- Uses ReLU activations and one shared dropout probability.
- Returns raw logits (no softmax) for use with CrossEntropyLoss.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron for flattened image inputs.

    Args:
        input_dim: Flattened input dimension (e.g., 784 for MNIST, 3072 for CIFAR-10).
        hidden_dims: Sizes of hidden layers, e.g. [512, 256, 128].
        dropout: Dropout probability applied after each hidden ReLU activation.
        num_classes: Number of output classes. Defaults to 10.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.2,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer size.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError("All hidden layer sizes in hidden_dims must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape (batch_size, num_classes)."""
        x = torch.flatten(x, start_dim=1)
        x = self.hidden(x)
        logits = self.output(x)
        return logits
