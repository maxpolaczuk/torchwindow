from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """GPU-side half of a Window: registers a presentation texture and
    uploads tensor data into it each frame. Window owns the SDL window,
    GL context, shader program, and event loop; Backend owns the
    GPU-API-specific state (e.g. cudaGraphicsResource, MTLTexture)."""

    @classmethod
    @abstractmethod
    def supports(cls, tensor: Any) -> bool:
        """Return True if this backend can upload ``tensor`` directly."""

    @abstractmethod
    def register(self, tex: int, width: int, height: int) -> None:
        """Bind the backend to the given GL texture and size. Called once
        per Window lifecycle and again on resize."""

    @abstractmethod
    def upload(
        self,
        tensor: Any,
        width: int,
        height: int,
        stream: Any = None,
    ) -> None:
        """Copy tensor contents into the registered texture."""

    @abstractmethod
    def unregister(self) -> None:
        """Release any GPU-API-specific handles. Idempotent."""
