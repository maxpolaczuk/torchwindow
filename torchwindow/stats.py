from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClientMetrics:
    client_id: str
    width: int
    height: int
    fps: float
    encode_ms_avg: float = 0.0
    encode_samples: int = 0
    rtt_ms: Optional[float] = None
    frames_dropped: int = 0
    forced_keyframes: int = 0
    last_idr_reason: Optional[str] = None

    def update_encode(self, duration_ms: float) -> None:
        # Exponential moving average with emphasis on latest samples.
        if self.encode_samples == 0:
            self.encode_ms_avg = duration_ms
        else:
            self.encode_ms_avg = 0.7 * self.encode_ms_avg + 0.3 * duration_ms
        self.encode_samples += 1

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "id": self.client_id,
            "w": self.width,
            "h": self.height,
            "fps": round(self.fps, 2),
            "encode_ms": round(self.encode_ms_avg, 2),
            "frames_dropped": self.frames_dropped,
            "forced_idr": self.forced_keyframes,
        }
        if self.rtt_ms is not None:
            data["rtt_ms"] = round(self.rtt_ms, 2)
        if self.last_idr_reason:
            data["last_idr_reason"] = self.last_idr_reason
        return data


class StatsCollector:
    """Collects ingest and per-client streaming metrics."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._frame_times: deque[float] = deque(maxlen=240)
        self._queue_depth: int = 0
        self._dropped_total: int = 0
        self._clients: Dict[str, ClientMetrics] = {}
        self._encoder_mode: str = "unknown"
        self._encoder_detail: Optional[str] = None

    async def record_frame_ingest(self, queue_depth: int, dropped: int) -> None:
        async with self._lock:
            self._frame_times.append(time.perf_counter())
            self._queue_depth = queue_depth
            self._dropped_total = dropped

    async def register_client(
        self, client_id: str, width: int, height: int, fps: float
    ) -> None:
        async with self._lock:
            self._clients[client_id] = ClientMetrics(
                client_id=client_id, width=width, height=height, fps=fps
            )

    async def unregister_client(self, client_id: str) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)

    async def record_encode(self, client_id: str, duration_ms: float) -> None:
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.update_encode(duration_ms)

    async def record_rtt(self, client_id: str, rtt_ms: float) -> None:
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.rtt_ms = rtt_ms

    async def record_dropped(self, client_id: str, count: int) -> None:
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.frames_dropped = count

    async def record_forced_idr(self, client_id: str, reason: str) -> None:
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.forced_keyframes += 1
                client.last_idr_reason = reason

    async def update_viewport(self, client_id: str, width: int, height: int, fps: float) -> None:
        async with self._lock:
            client = self._clients.get(client_id)
            if client:
                client.width = width
                client.height = height
                client.fps = fps

    async def snapshot(self, stream_name: str) -> Dict[str, object]:
        async with self._lock:
            fps = self._compute_fps()
            clients: List[Dict[str, object]] = [
                client.to_dict() for client in self._clients.values()
            ]
            return {
                "name": stream_name,
                "server_fps": fps,
                "encode_ms_avg": self._aggregate_encode(),
                "queue_depth": self._queue_depth,
                "dropped_frames": self._dropped_total,
                "subscribers": len(self._clients),
                "per_client": clients,
                "encoder_mode": self._encoder_mode,
                "encoder_detail": self._encoder_detail,
            }

    def _compute_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        elapsed = self._frame_times[-1] - self._frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._frame_times) - 1) / elapsed

    def _aggregate_encode(self) -> float:
        total = 0.0
        count = 0
        for client in self._clients.values():
            if client.encode_samples:
                total += client.encode_ms_avg
                count += 1
        return round(total / count, 2) if count else 0.0

    async def set_encoder_mode(self, mode: str, detail: Optional[str] = None) -> None:
        async with self._lock:
            self._encoder_mode = mode
            self._encoder_detail = detail

    @property
    def encoder_mode(self) -> str:
        return self._encoder_mode

    @property
    def encoder_detail(self) -> Optional[str]:
        return self._encoder_detail
