from __future__ import annotations

import logging
from typing import Any

from cuda import cudart as cu
from OpenGL import GL as gl

from ..exceptions import CudaException, OpenGLException
from .base import Backend

logger = logging.getLogger(__name__)

_BYTES_PER_PIXEL = 4 * 4  # RGBA float32


class CudaGLBackend(Backend):
    """Zero-copy PyTorch-CUDA → OpenGL texture via cudaGraphicsGLRegisterImage."""

    def __init__(self) -> None:
        self.cuda_image = None

    @classmethod
    def supports(cls, tensor: Any) -> bool:
        device = getattr(tensor, "device", None)
        return device is not None and getattr(device, "type", None) == "cuda"

    def register(self, tex: int, width: int, height: int) -> None:
        self.unregister()
        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise OpenGLException(
                "OpenGL context may be running on integrated graphics"
            )
        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to register opengl texture")

    def upload(
        self,
        tensor: Any,
        width: int,
        height: int,
        stream: Any = None,
    ) -> None:
        if self.cuda_image is None:
            raise RuntimeError("CudaGLBackend used before register()")

        cu_stream = self._resolve_stream(stream)
        pitch = _BYTES_PER_PIXEL * width

        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu_stream)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to map graphics resource")
        try:
            err, array = cu.cudaGraphicsSubResourceGetMappedArray(
                self.cuda_image, 0, 0
            )
            if err != cu.cudaError_t.cudaSuccess:
                raise CudaException("Unable to get mapped array")
            (err,) = cu.cudaMemcpy2DToArrayAsync(
                array,
                0,
                0,
                tensor.data_ptr(),
                pitch,
                pitch,
                height,
                cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                cu_stream,
            )
            if err != cu.cudaError_t.cudaSuccess:
                raise CudaException("Unable to copy from tensor to texture")
        finally:
            (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu_stream)
            if err != cu.cudaError_t.cudaSuccess:
                raise CudaException("Unable to unmap graphics resource")

    def unregister(self) -> None:
        if self.cuda_image is None:
            return
        (err,) = cu.cudaGraphicsUnregisterResource(self.cuda_image)
        if err != cu.cudaError_t.cudaSuccess:
            logger.warning("cudaGraphicsUnregisterResource returned %s", err)
        self.cuda_image = None

    @staticmethod
    def _resolve_stream(stream: Any) -> int:
        if stream is None:
            return cu.cudaStreamLegacy
        ptr = getattr(stream, "cuda_stream", None)
        if ptr is not None:
            return ptr
        return stream
