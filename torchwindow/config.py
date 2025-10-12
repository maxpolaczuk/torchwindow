from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_STUN = (
    "stun:stun.l.google.com:19302",
    # "stun:stun1.l.google.com:19302",
    # "stun:stun2.l.google.com:19302",
    # "stun:stun3.l.google.com:19302",
    # "stun:stun4.l.google.com:19302",
    # "stun:stun5.l.google.com:19302",
)


@dataclass
class StreamCaps:
    max_width: int
    max_height: int
    max_fps: int

    def clamp(self, width: int, height: int, fps: int) -> Tuple[int, int, int]:
        """Clamp viewport parameters to the negotiated caps."""
        return (
            min(width, self.max_width),
            min(height, self.max_height),
            min(fps, self.max_fps),
        )


@dataclass
class WindowConfig:
    width: int = 1280
    height: int = 720
    name: str = "TorchWindow 2.0"
    stream: bool = True
    sfu_mode: str = "embedded"  # "embedded" | "remote"
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    stun_servers: Sequence[str] = field(default_factory=lambda: list(DEFAULT_STUN))
    turn_servers: Sequence[str] = field(default_factory=list)
    lossless: bool = True
    codec: str = "h264"
    prefer_hardware_encode: bool = True
    sfu_url: Optional[str] = None
    public_url: Optional[str] = None
    max_viewport_width: int = 1920
    max_viewport_height: int = 1080
    max_viewport_fps: int = 60
    watchdog_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.lossless:
            raise ValueError("TorchWindow 2.0 enforces lossless=True")
        if self.sfu_mode not in {"embedded", "remote"}:
            raise ValueError(f"Unsupported sfu_mode={self.sfu_mode!r}")
        if self.codec not in {"av1", "hevc", "h264"}:
            raise ValueError("codec must be 'av1', 'hevc', or 'h264'")
        if self.sfu_mode == "remote" and not self.sfu_url:
            raise ValueError("sfu_url is required when sfu_mode='remote'")

    @property
    def stream_caps(self) -> StreamCaps:
        return StreamCaps(
            max_width=self.max_viewport_width,
            max_height=self.max_viewport_height,
            max_fps=self.max_viewport_fps,
        )

    def stun_list(self) -> List[str]:
        return list(self.stun_servers)

    def turn_list(self) -> List[str]:
        return list(self.turn_servers)

    def with_caps(self, caps: StreamCaps) -> "WindowConfig":
        clone = dataclass_replace(self)
        clone.max_viewport_width = caps.max_width
        clone.max_viewport_height = caps.max_height
        clone.max_viewport_fps = caps.max_fps
        return clone


def dataclass_replace(config: WindowConfig, **changes: object) -> WindowConfig:
    """Return a copy of the config with selected fields replaced."""
    params = config.__dict__.copy()
    params.update(changes)
    return WindowConfig(**params)
