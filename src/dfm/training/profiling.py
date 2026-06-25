"""Lightweight profiling helpers."""

from __future__ import annotations


def count_parameters(model, trainable_only: bool = True) -> int:
    """Count model parameters for PyTorch-like modules."""

    parameters = model.parameters()
    if trainable_only:
        parameters = (param for param in parameters if param.requires_grad)
    return sum(param.numel() for param in parameters)

