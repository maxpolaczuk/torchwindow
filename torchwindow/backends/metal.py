"""Metal backend: PyTorch-MPS tensor -> MTLTexture.

V1 strategy (CPU roundtrip)
---------------------------
PyTorch-MPS stores tensors in private-storage MTLBuffers. Crucially,
``tensor.data_ptr()`` on MPS is *not* a CPU-readable address — even
on Apple Silicon's unified memory architecture, the pointer points
into a region only the GPU can dereference (verified empirically on
torch 2.7.1). So the ideal zero-copy path (wrap the tensor's backing
MTLBuffer and create an MTLTexture view over it) requires access to
that MTLBuffer handle, which PyTorch does not expose in its public
Python API.

V1 therefore does a single CPU roundtrip per frame:
``tensor.cpu().contiguous().numpy()`` → memmove into a shared-storage
MTLBuffer that backs our MTLTexture. This is a real CPU↔GPU transfer
(unlike the CUDA path's device-to-device copy), but proves the render
pipeline end-to-end and lets users see their tensors immediately.

Follow-ups for true zero-copy
-----------------------------
Two viable routes, each a separate PR:

1. Reach into PyTorch's C++ backing via a private selector like
   ``at::mps::getMTLBufferStorage(tensor)`` exposed through a tiny
   C++ extension. Then ``newTextureWithDescriptor:offset:bytesPerRow:``
   on that buffer gives a zero-copy MTLTexture view.

2. Keep the public API but use a GPU-side blit: allocate a
   shared-storage MTLBuffer, capture PyTorch's MTLCommandQueue (via
   ``torch.mps.current_stream().synchronize()``), submit an MTLBlit
   encoder copying from the tensor buffer to ours. No CPU involvement
   but still needs the private MTLBuffer handle.

Sync
----
Before copying we call ``torch.mps.synchronize()`` to flush in-flight
MPS kernels so the tensor contents are stable.

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

        # V1: CPU roundtrip. tensor.data_ptr() on MPS is GPU-only and
        # not dereferenceable from the CPU, so we force a host copy.
        # Numpy's C buffer is the source; MTLBuffer.contents() is the
        # destination. Both writes are seen by the GPU next frame
        # because MTLResourceStorageModeShared is CPU/GPU-coherent on
        # Apple Silicon.
        nbytes = pitch * height
        host = tensor.detach().cpu().contiguous().numpy()
        src_ptr = host.ctypes.data

        dst_buffer = self._buffer.contents().as_buffer(length)
        dst_arr = (ctypes.c_ubyte * length).from_buffer(dst_buffer)
        ctypes.memmove(dst_arr, src_ptr, nbytes)
        self._last_data_ptr = src_ptr
        # Keep ``host`` alive until after memmove — numpy may free its
        # storage otherwise. The local reference above is sufficient.

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
