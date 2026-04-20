from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["NotebookWindow", "Window"]

if TYPE_CHECKING:
    from .notebook import NotebookWindow
    from .window import Window


def __getattr__(name: str) -> Any:
    if name == "Window":
        from .window import Window as _Window

        return _Window
    if name == "NotebookWindow":
        from .notebook import NotebookWindow as _NotebookWindow

        return _NotebookWindow
    raise AttributeError(f"module 'torchwindow' has no attribute {name!r}")
