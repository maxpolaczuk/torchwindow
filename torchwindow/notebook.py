"""Jupyter ``NotebookWindow`` — live tensor viewer via anywidget.

Usage
-----
    from torchwindow import NotebookWindow
    import torch

    w = NotebookWindow(512, 512)
    w                              # displays widget inline
    for _ in range(100):
        frame = torch.rand(512, 512, 4, device="mps")
        w.draw(frame)

Design
------
The browser can't read GPU memory directly (no stable WebGPU-interop
for PyTorch yet), so the server-side hop is unavoidable:
``tensor.cpu().numpy()`` -> convert to ``uint8`` RGBA -> set a
``traitlets.Bytes`` trait that anywidget's protocol syncs to the JS
side. The client ``putImageData``s into a ``<canvas>``. That's it.

The class accepts the same ``(H, W, 4) float32`` tensors as the native
``Window`` (via ``_validate.validate_tensor``); Phase 3 loosens this
to uint8 / (H,W,3) / (C,H,W).

Performance
-----------
uint8 RGBA at 1080p is ~8 MB/frame, which Jupyter's websocket can push
at 20-40 fps depending on kernel distance. For faster needs we have
two levers:
- send a quality-reduced JPEG/WebP via ``imageio`` or ``Pillow``
- render offscreen with EGL/Metal and send a downsampled thumbnail

Neither is needed for V1 (RGBA8 is the simplest correct path).

Import shape
------------
``anywidget`` is an optional dep (``pip install torchwindow[notebook]``).
This module imports it unconditionally because it only loads when the
user explicitly asks for ``NotebookWindow``.
"""

from __future__ import annotations

from typing import Any

import anywidget
import numpy as np
import traitlets

from ._normalize import to_canonical
from ._validate import validate_tensor

_ESM = """
function render({ model, el }) {
  const canvas = document.createElement('canvas');
  canvas.style.maxWidth = '100%';
  canvas.style.imageRendering = 'pixelated';
  canvas.style.border = '1px solid #444';
  el.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  function paint() {
    const w = model.get('width');
    const h = model.get('height');
    const buf = model.get('frame_bytes');
    if (!w || !h || !buf || buf.byteLength === 0) return;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    const bytes = buf instanceof ArrayBuffer ? new Uint8Array(buf) : new Uint8Array(buf.buffer);
    const clamped = new Uint8ClampedArray(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const img = new ImageData(clamped, w, h);
    ctx.putImageData(img, 0, 0);
  }

  model.on('change:frame_bytes', paint);
  paint();
}
export default { render };
"""


def _float_rgba_to_uint8(arr: np.ndarray) -> bytes:
    """(H, W, 4) float32 in [0,1] -> uint8 RGBA bytes, tight packing."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.tobytes()


class NotebookWindow(anywidget.AnyWidget):
    """``Window``-compatible viewer that renders into a Jupyter canvas."""

    _esm = _ESM
    width = traitlets.Int(800).tag(sync=True)
    height = traitlets.Int(600).tag(sync=True)
    frame_bytes = traitlets.Bytes(b"").tag(sync=True)

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        name: str = "torchwindow",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.name = name
        self.running = True

    # Parity with Window -- matches the Window API used by example.py
    # and user code alike.
    def draw(self, tensor: Any, stream: Any = None) -> None:  # noqa: ARG002
        if not self.running:
            return
        tensor = to_canonical(tensor)
        validate_tensor(tensor, self.width, self.height)
        arr = tensor.detach().cpu().contiguous().numpy()
        self.frame_bytes = _float_rgba_to_uint8(arr)

    def close(self) -> None:
        self.running = False
        self.frame_bytes = b""

    def __enter__(self) -> "NotebookWindow":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
