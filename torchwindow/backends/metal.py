"""Metal backend: PyTorch-MPS tensor -> MTLTexture.

V1 strategy
-----------
PyTorch-MPS allocates its tensors inside MTLBuffers with shared
storage. ``tensor.data_ptr()`` is a pointer into that unified-memory
region. Ideally we would wrap it with
``newBufferWithBytesNoCopy:length:options:deallocator:`` for a true
zero-copy path — but pyobjc's signature inspector treats the first
argument as a Python buffer-protocol object of the ctypes pointer's
own size (8 bytes) rather than the underlying memory region. Calling
the selector via ctypes + ``objc_msgSend`` works but is a significant
detour; we take it as a follow-up.

V1 allocates a shared-storage MTLBuffer once, then uses
``ctypes.memmove`` each frame to copy tensor bytes into the buffer's
``contents`` pointer. Since both source and destination live in
unified memory on Apple Silicon, this is a GPU-visible region-to-region
copy at memcpy speed — still no CPU↔GPU transfer, just not yet zero
MTLBuffer-wrap-free. Once the v1 render pipeline is proven end-to-end
we swap this for the no-copy path.

Sync
----
Before we sample the texture in the render pass we call
``torch.mps.synchronize()``. This is coarse — a future pass can
share an MTLCommandQueue with PyTorch via
``torch.mps.current_stream()`` and insert an MTLSharedEvent fence
instead — but for a first pass it guarantees correctness.

Interaction with Window
-----------------------
Unlike CudaGLBackend (which writes into a GL texture owned by the
Window), MetalBackend owns the MTLTexture itself. The MetalWindow
hands us its MTLDevice on register() and fetches the current texture
via ``current_texture()`` when building a render pass.
"""

from __future__ import annotations

import ctypes
from typing import Any, Optional

from ..exceptions import CudaException  # reused for raise-on-mps-failure path
from .base import Backend

_BYTES_PER_PIXEL = 4 * 4  # RGBA float32
_PAGE_SIZE = 16 * 1024  # Apple Silicon page size


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


class MetalBackend(Backend):
    """PyTorch-MPS tensor -> MTLTexture zero-copy view.

    The instance is lazily initialized: ``register(device, width, height)``
    receives the MTLDevice from the owning MetalWindow (not a GL texture
    id like CudaGLBackend), because the backend owns the MTLTexture.
    """

    def __init__(self) -> None:
        self.device: Any = None
        self.texture: Any = None
        self._buffer: Any = None
        self._width = 0
        self._height = 0
        self._last_data_ptr: int = 0

    @classmethod
    def supports(cls, tensor: Any) -> bool:
        device = getattr(tensor, "device", None)
        return device is not None and getattr(device, "type", None) == "mps"

    def register(self, device: Any, width: int, height: int) -> None:
        """``device`` is an MTLDevice object, not a GL texture id.

        MetalWindow passes its device here; Window.Backend's signature
        is ``register(tex, width, height)`` so we keep the name but
        accept a device for this backend. The MetalWindow code path is
        the only caller.
        """
        self.device = device
        self._width = width
        self._height = height
        self._buffer = None
        self.texture = None
        self._last_data_ptr = 0

    def current_texture(self) -> Any:
        """Return the MTLTexture view for the most recently uploaded tensor.

        MetalWindow.render() binds this to fragment texture slot 0.
        """
        return self.texture

    def upload(
        self,
        tensor: Any,
        width: int,
        height: int,
        stream: Any = None,
    ) -> None:
        if self.device is None:
            raise RuntimeError("MetalBackend used before register()")

        import Metal  # type: ignore[import-not-found]

        try:
            import torch  # type: ignore[import-not-found]

            torch.mps.synchronize()
        except ImportError:
            pass  # caller is responsible for sync if torch isn't importable

        pitch = _BYTES_PER_PIXEL * width
        length = _round_up(pitch * height, _PAGE_SIZE)

        # One-time: allocate the shared MTLBuffer + MTLTexture view.
        if self._buffer is None:
            self._buffer = self.device.newBufferWithLength_options_(
                length, Metal.MTLResourceStorageModeShared
            )
            if self._buffer is None:
                raise CudaException("newBufferWithLength:options: returned nil")

            desc = Metal.MTLTextureDescriptor.alloc().init()
            desc.setPixelFormat_(Metal.MTLPixelFormatRGBA32Float)
            desc.setWidth_(width)
            desc.setHeight_(height)
            desc.setUsage_(Metal.MTLTextureUsageShaderRead)
            desc.setStorageMode_(Metal.MTLStorageModeShared)

            self.texture = self._buffer.newTextureWithDescriptor_offset_bytesPerRow_(
                desc, 0, pitch
            )
            if self.texture is None:
                raise CudaException(
                    "newTextureWithDescriptor:offset:bytesPerRow: returned nil"
                )

        # Per-frame copy: tensor bytes -> MTLBuffer.contents(). Both
        # regions are in unified memory; this is a plain memcpy, no
        # CPU↔GPU transfer. ~1-2ms for 800x600 RGBA32F.
        nbytes = pitch * height
        # pyobjc exposes MTLBuffer.contents() as an objc.varlist; its
        # as_buffer(n) returns a writable Python buffer pointing at the
        # same memory. We wrap it as a ctypes array to get an address
        # for memmove.
        dst_buffer = self._buffer.contents().as_buffer(length)
        dst_arr = (ctypes.c_ubyte * length).from_buffer(dst_buffer)
        src_ptr = int(tensor.data_ptr())
        ctypes.memmove(dst_arr, src_ptr, nbytes)
        self._last_data_ptr = src_ptr

    def unregister(self) -> None:
        self.texture = None
        self._buffer = None
        self.device = None
        self._last_data_ptr = 0


def _try_register() -> Optional[type]:
    try:
        import Metal  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return None
    return MetalBackend
