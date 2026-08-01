"""
Shared device resolution for the torch-based grouping methods (AlertBERT,
DeepCASE). Checks mps before cuda -- matches the fuller auto-detect chain
already used by thesis.baselines.cscas_bert/cscas_securebert (the grouping
methods previously only checked cuda->cpu).
"""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """
    None or "auto" -> mps if available, else cuda if available, else cpu.
    Anything else is passed straight to torch.device(...) as an explicit
    override (e.g. "cpu", "cuda:1").
    """
    if isinstance(device, torch.device):
        return device
    if device and device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
