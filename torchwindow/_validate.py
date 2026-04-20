from __future__ import annotations

from typing import Any


def validate_tensor(tensor: Any, width: int, height: int) -> None:
    """Raise if ``tensor`` is not a contiguous float32 (H, W, 4) CUDA tensor
    matching ``(height, width)``. Kept as a module-level function so tests
    can import it without pulling in SDL/GL."""
    if not hasattr(tensor, "data_ptr") or not hasattr(tensor, "shape"):
        raise TypeError(
            f"Expected a torch.Tensor, got {type(tensor).__name__}"
        )
    shape = tuple(tensor.shape)
    if len(shape) != 3 or shape[2] != 4:
        raise ValueError(f"Expected tensor shape (H, W, 4), got {shape}")
    h, w, _ = shape
    if h != height or w != width:
        raise ValueError(
            f"Tensor (H={h}, W={w}) does not match window "
            f"({height}, {width})"
        )
    dtype = getattr(tensor, "dtype", None)
    if dtype is not None and str(dtype) != "torch.float32":
        raise ValueError(f"Expected dtype torch.float32, got {dtype}")
    is_contig = getattr(tensor, "is_contiguous", None)
    if callable(is_contig) and not is_contig():
        raise ValueError(
            "Tensor must be contiguous; call tensor.contiguous() first."
        )
