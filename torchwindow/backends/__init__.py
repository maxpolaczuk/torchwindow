from __future__ import annotations

from typing import Any, List, Optional, Type

from .base import Backend

_REGISTRY: List[Type[Backend]] = []


def register_backend(cls: Type[Backend]) -> Type[Backend]:
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)
    return cls


def _try_register_cuda_gl() -> None:
    try:
        from .cuda_gl import CudaGLBackend
    except Exception:  # missing cuda-python / no CUDA driver
        return
    register_backend(CudaGLBackend)


def _try_register_metal() -> None:
    try:
        from .metal import MetalBackend
    except Exception:  # missing pyobjc-framework-Metal / non-macOS
        return
    register_backend(MetalBackend)


_try_register_cuda_gl()
_try_register_metal()


def auto_select(tensor: Any) -> Backend:
    for cls in _REGISTRY:
        if cls.supports(tensor):
            return cls()
    devices = _describe_registered()
    raise TypeError(
        f"No registered backend supports tensor on device "
        f"{getattr(tensor, 'device', '<unknown>')}. "
        f"Registered backends: {devices or 'none'}"
    )


def resolve(name: Optional[str], tensor: Any) -> Backend:
    if name is None:
        return auto_select(tensor)
    for cls in _REGISTRY:
        if cls.__name__.lower().replace("backend", "") == name.lower():
            return cls()
    raise ValueError(
        f"Unknown backend '{name}'. Available: {_describe_registered()}"
    )


def _describe_registered() -> str:
    return ", ".join(c.__name__ for c in _REGISTRY)


__all__ = ["Backend", "auto_select", "resolve", "register_backend"]
