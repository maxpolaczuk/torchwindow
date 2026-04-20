"""Public ``Window`` factory.

Two presentation paths exist:
- ``windows.gl_window.GLWindow`` — SDL2 + OpenGL, paired with
  ``backends.cuda_gl.CudaGLBackend``. Linux / Windows (NVIDIA).
- ``windows.metal_window.MetalWindow`` — SDL2 + Metal, paired with
  ``backends.metal.MetalBackend``. macOS (Apple Silicon / PyTorch MPS).

``Window(...)`` returns whichever implementation matches the requested
backend or the host platform. Both expose the same surface:
``.draw(tensor, stream=None)``, ``.close()``, context-manager protocol,
and a ``.running`` flag.
"""

from __future__ import annotations

import platform
from typing import Any, Optional


def _choose_implementation(backend: Optional[str]) -> str:
    if backend is not None:
        b = backend.lower()
        if b in ("cuda", "cudagl", "gl", "cuda_gl"):
            return "gl"
        if b in ("metal", "mps", "metalmps"):
            return "metal"
        raise ValueError(
            f"Unknown backend {backend!r}; expected 'cuda' or 'metal'"
        )
    # auto: macOS -> metal, everything else -> gl
    return "metal" if platform.system() == "Darwin" else "gl"


def Window(  # noqa: N802 — factory that mimics a class name
    width: int = 800,
    height: int = 600,
    name: str = "torchwindow",
    backend: Optional[str] = None,
) -> Any:
    impl = _choose_implementation(backend)
    if impl == "metal":
        from .windows.metal_window import MetalWindow

        return MetalWindow(width=width, height=height, name=name, backend=backend)
    from .windows.gl_window import GLWindow

    return GLWindow(width=width, height=height, name=name, backend=backend)
