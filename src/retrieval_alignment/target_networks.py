"""Tiny target-network utilities for DW-MQL smoke components.

These helpers provide the minimal mechanics needed before adding target critics
to the trainer: clone a frozen eval copy, hard-copy parameters and buffers, and
soft-update target parameters. They do not implement a training loop,
checkpointing, FAISS integration, or target refresh scheduling.
"""

from __future__ import annotations

import copy

try:  # pragma: no cover - exercised when dependencies are installed.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - keeps imports clean pre-install.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ModuleNotFoundError("PyTorch is required to use target-network utilities.")


def _freeze_eval(model: "nn.Module") -> "nn.Module":
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _validate_compatible(target_model: "nn.Module", source_model: "nn.Module") -> None:
    target_state = target_model.state_dict()
    source_state = source_model.state_dict()
    if target_state.keys() != source_state.keys():
        raise ValueError("target_model and source_model state_dict keys do not match.")
    for key, target_value in target_state.items():
        source_value = source_state[key]
        if target_value.shape != source_value.shape:
            raise ValueError(
                "target_model and source_model state_dict shapes do not match "
                f"for key {key!r}: {tuple(target_value.shape)} != {tuple(source_value.shape)}."
            )


def clone_target_network(model: "nn.Module") -> "nn.Module":
    """Return a frozen eval-mode deep copy of ``model``.

    The cloned target initially matches the source parameters exactly and all
    target parameters have ``requires_grad=False``.
    """

    _require_torch()
    target = copy.deepcopy(model)
    return _freeze_eval(target)


def hard_update_target(target_model: "nn.Module", source_model: "nn.Module") -> None:
    """Copy source parameters and buffers into a frozen target model."""

    _require_torch()
    _validate_compatible(target_model, source_model)
    with torch.no_grad():
        target_model.load_state_dict(source_model.state_dict())
    _freeze_eval(target_model)


def soft_update_target(target_model: "nn.Module", source_model: "nn.Module", tau: float) -> None:
    """Soft-update target parameters toward source parameters.

    ``target_param = (1 - tau) * target_param + tau * source_param``.
    Floating-point buffers are updated the same way; non-floating buffers are
    copied from the source when ``tau == 1`` and otherwise left unchanged.
    """

    _require_torch()
    if not isinstance(tau, int | float) or not 0.0 <= float(tau) <= 1.0:
        raise ValueError("tau must be in [0, 1].")
    tau_float = float(tau)
    _validate_compatible(target_model, source_model)

    with torch.no_grad():
        for target_param, source_param in zip(
            target_model.parameters(),
            source_model.parameters(),
            strict=True,
        ):
            target_param.mul_(1.0 - tau_float).add_(source_param, alpha=tau_float)

        for target_buffer, source_buffer in zip(
            target_model.buffers(),
            source_model.buffers(),
            strict=True,
        ):
            if torch.is_floating_point(target_buffer):
                target_buffer.mul_(1.0 - tau_float).add_(source_buffer, alpha=tau_float)
            elif tau_float == 1.0:
                target_buffer.copy_(source_buffer)

    _freeze_eval(target_model)
