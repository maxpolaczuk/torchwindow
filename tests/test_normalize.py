"""Tensor normalizer tests.

Exercise the shape + dtype paths using real CPU torch tensors. No GPU
required — the normalizer is device-agnostic.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torchwindow._normalize import to_canonical  # noqa: E402


def _ones_hwc(h: int, w: int, c: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.ones((h, w, c), dtype=dtype)


def test_identity_hwc4_float32() -> None:
    t = _ones_hwc(4, 5, 4, torch.float32)
    out = to_canonical(t)
    assert out.shape == (4, 5, 4)
    assert out.dtype == torch.float32


def test_hwc3_pads_alpha() -> None:
    t = torch.tensor([[[0.25, 0.5, 0.75]]], dtype=torch.float32)  # (1,1,3)
    out = to_canonical(t)
    assert out.shape == (1, 1, 4)
    assert out[0, 0].tolist() == [0.25, 0.5, 0.75, 1.0]


def test_hwc1_expands_to_rgba() -> None:
    t = torch.full((2, 2, 1), 0.5, dtype=torch.float32)
    out = to_canonical(t)
    assert out.shape == (2, 2, 4)
    # greyscale replicated across R/G/B, alpha=1
    assert out[0, 0].tolist() == [0.5, 0.5, 0.5, 1.0]


def test_chw_permutes_to_hwc() -> None:
    t = torch.zeros((3, 4, 5), dtype=torch.float32)
    t[0] = 1.0  # red channel full
    out = to_canonical(t)
    assert out.shape == (4, 5, 4)
    assert out[0, 0, 0].item() == 1.0
    assert out[0, 0, 1].item() == 0.0
    assert out[0, 0, 3].item() == 1.0  # alpha padded


def test_uint8_rescales_to_unit_float() -> None:
    t = torch.full((2, 2, 4), 255, dtype=torch.uint8)
    out = to_canonical(t)
    assert out.dtype == torch.float32
    assert out[0, 0].tolist() == [1.0, 1.0, 1.0, 1.0]

    t0 = torch.zeros((2, 2, 4), dtype=torch.uint8)
    out0 = to_canonical(t0)
    assert out0[0, 0, 0].item() == 0.0


def test_float16_upcasts() -> None:
    t = torch.full((2, 2, 4), 0.25, dtype=torch.float16)
    out = to_canonical(t)
    assert out.dtype == torch.float32
    assert abs(out[0, 0, 0].item() - 0.25) < 1e-4


def test_rejects_weird_shape() -> None:
    t = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="3-D tensor"):
        to_canonical(t)


def test_rejects_unsupported_channel_count() -> None:
    # Last dim is 5, first dim is 7 — neither matches {1,3,4}
    t = torch.zeros((7, 3, 5), dtype=torch.float32)
    with pytest.raises(ValueError, match="Cannot infer layout"):
        to_canonical(t)


def test_rejects_unsupported_dtype() -> None:
    t = torch.zeros((2, 2, 4), dtype=torch.int32)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        to_canonical(t)
