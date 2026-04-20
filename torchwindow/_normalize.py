"""Tensor shape/dtype normalizer.

Every backend (CUDA-GL, Metal, NotebookWindow) expects a
``(H, W, 4) float32`` tensor, so we normalize *once* up front and let
validation check only the canonical form. Normalization stays on the
tensor's current device: (C,H,W)->(H,W,C) is a contiguous permute,
alpha-pad is a ``torch.cat`` with ones, uint8->float32 is a ``.to``
with scale — all GPU ops when the input is on GPU, keeping the
transfer-free story intact.

Accepted inputs
---------------
Shape: ``(H, W, 4)``, ``(H, W, 3)``, ``(H, W, 1)``, ``(C, H, W)`` with
``C in {1, 3, 4}``. The ambiguous ``(H, W, 3)`` vs ``(3, H, W)`` case
is disambiguated by: if the last dim is 1/3/4, we treat as HWC; else
if the first dim is 1/3/4, we treat as CHW. Tensors where neither
leading nor trailing dim is in {1,3,4} raise.

Dtype: ``torch.uint8`` (scaled by 1/255), ``torch.float16`` / ``bfloat16``
(upcast), ``torch.float32`` (pass-through). Other dtypes raise.

``np.ndarray`` and anything else without a ``.to`` method is returned
unchanged; the native backends will reject it via
``_validate.validate_tensor``.
"""

from __future__ import annotations

from typing import Any

_CANON_CHANNELS = 4


def to_canonical(tensor: Any) -> Any:
    """Return a (H, W, 4) float32 tensor on the input's device."""
    to = getattr(tensor, "to", None)
    shape_attr = getattr(tensor, "shape", None)
    if to is None or shape_attr is None:
        return tensor

    t = tensor
    shape = tuple(t.shape)
    if len(shape) != 3:
        raise ValueError(
            f"Expected 3-D tensor (H,W,C) or (C,H,W), got shape {shape}"
        )

    # Shape: normalize to HWC
    last, first = shape[-1], shape[0]
    if last in (1, 3, 4):
        pass  # already HWC
    elif first in (1, 3, 4) and last not in (1, 3, 4):
        t = t.permute(1, 2, 0).contiguous()
    else:
        raise ValueError(
            f"Cannot infer layout from shape {shape}; expected a dim of "
            "size 1, 3, or 4 for channels."
        )

    # Dtype: normalize to float32 in [0,1]
    import torch

    if t.dtype == torch.uint8:
        t = t.to(torch.float32).mul_(1.0 / 255.0)
    elif t.dtype in (torch.float16, torch.bfloat16):
        t = t.to(torch.float32)
    elif t.dtype != torch.float32:
        raise ValueError(
            f"Unsupported dtype {t.dtype}; expected uint8, float16, "
            "bfloat16, or float32."
        )

    # Channels: pad/expand to 4
    c = t.shape[-1]
    if c == 1:
        t = t.expand(t.shape[0], t.shape[1], 3).contiguous()
        c = 3
    if c == 3:
        alpha = torch.ones(
            (t.shape[0], t.shape[1], 1), dtype=t.dtype, device=t.device
        )
        t = torch.cat([t, alpha], dim=-1)
    elif c != _CANON_CHANNELS:
        raise ValueError(
            f"Unsupported channel count {c}; expected 1, 3, or 4."
        )

    return t.contiguous()
