from __future__ import annotations

from types import SimpleNamespace

import pytest

from torchwindow._validate import validate_tensor


class FakeTensor:
    def __init__(
        self,
        shape,
        dtype: str = "torch.float32",
        device_type: str = "cuda",
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type)
        self._contig = contiguous

    def is_contiguous(self) -> bool:
        return self._contig

    def data_ptr(self) -> int:
        return 0


def test_accepts_valid_tensor() -> None:
    validate_tensor(FakeTensor((600, 800, 4)), width=800, height=600)


def test_rejects_non_tensor() -> None:
    with pytest.raises(TypeError, match="Expected a torch.Tensor"):
        validate_tensor(object(), width=800, height=600)


def test_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match=r"shape \(H, W, 4\)"):
        validate_tensor(FakeTensor((600, 800)), width=800, height=600)


def test_rejects_wrong_channels() -> None:
    with pytest.raises(ValueError, match=r"shape \(H, W, 4\)"):
        validate_tensor(FakeTensor((600, 800, 3)), width=800, height=600)


def test_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match window"):
        validate_tensor(FakeTensor((480, 640, 4)), width=800, height=600)


def test_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="Expected dtype torch.float32"):
        validate_tensor(
            FakeTensor((600, 800, 4), dtype="torch.uint8"),
            width=800,
            height=600,
        )


def test_rejects_non_contiguous() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        validate_tensor(
            FakeTensor((600, 800, 4), contiguous=False),
            width=800,
            height=600,
        )
