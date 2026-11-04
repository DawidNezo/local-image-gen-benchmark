"""Optional PyTorch import.

PyTorch is needed to generate images, but `validate`, `devices` and `report` must still start
without it, so every module takes `torch` from here and handles `None`.
"""

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at runtime
    torch = None

__all__ = ["torch"]
