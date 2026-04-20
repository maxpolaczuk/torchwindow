from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["Window"]

if TYPE_CHECKING:
    from .window import Window


def __getattr__(name: str) -> Any:
    if name == "Window":
        from .window import Window as _Window

        return _Window
    raise AttributeError(f"module 'torchwindow' has no attribute {name!r}")
