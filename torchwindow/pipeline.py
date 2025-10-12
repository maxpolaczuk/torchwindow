from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .stats import StatsCollector


@dataclass
class FrameRecord:
    sequence: int
    tensor: torch.Tensor
    timestamp: float


@dataclass
class ViewportState:
    view_id: str = "default"
    center_px: Tuple[int, int] = (0, 0)
    zoom: float = 1.0
    target_size: Tuple[int, int] = (1280, 720)
    fps_cap: int = 60


class FrameIngest:
    """Lock-free ring buffer (async friendly) that keeps the latest frames."""

    def __init__(self, depth: int = 2) -> None:
        if depth < 1:
            raise ValueError("FrameIngest depth must be >= 1")
        self._frames: deque[FrameRecord] = deque(maxlen=depth)
        self._sequence = 0
        self._dropped = 0
        self._condition = asyncio.Condition()

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def depth(self) -> int:
        return len(self._frames)

    @property
    def latest_sequence(self) -> int:
        return self._frames[-1].sequence if self._frames else 0

    async def push(self, tensor: torch.Tensor) -> FrameRecord:
        now = time.perf_counter()
        async with self._condition:
            self._sequence += 1
            sequence = self._sequence
            if len(self._frames) == self._frames.maxlen:
                self._frames.popleft()
                self._dropped += 1
            record = FrameRecord(sequence=sequence, tensor=tensor, timestamp=now)
            self._frames.append(record)
            self._condition.notify_all()
            return record

    async def wait_for_next(self, after_sequence: int) -> FrameRecord:
        async with self._condition:
            while True:
                if self._frames and self._frames[-1].sequence > after_sequence:
                    return self._frames[-1]
                await self._condition.wait()


class ViewportComposer:
    def __init__(self, width: int, height: int) -> None:
        self._base_width = width
        self._base_height = height

    def compose(self, frame: torch.Tensor, viewport: ViewportState) -> torch.Tensor:
        """Crop and scale the frame based on the viewport definition."""
        tensor = self._ensure_tensor(frame)
        base_h, base_w = tensor.shape[0], tensor.shape[1]
        cx, cy = viewport.center_px
        zoom = max(viewport.zoom, 1.0)

        crop_w = int(round(base_w / zoom))
        crop_h = int(round(base_h / zoom))
        crop_w = max(1, min(base_w, crop_w))
        crop_h = max(1, min(base_h, crop_h))

        # Clamp center coordinates to valid area.
        cx = int(max(crop_w // 2, min(base_w - crop_w // 2, cx)))
        cy = int(max(crop_h // 2, min(base_h - crop_h // 2, cy)))

        x0 = cx - crop_w // 2
        y0 = cy - crop_h // 2
        x1 = x0 + crop_w
        y1 = y0 + crop_h

        cropped = tensor[y0:y1, x0:x1, :]
        target_w, target_h = viewport.target_size
        if (target_w, target_h) == (cropped.shape[1], cropped.shape[0]):
            return cropped.contiguous()

        resized = self._resize(cropped, target_w, target_h)
        return resized

    def _resize(self, tensor: torch.Tensor, width: int, height: int) -> torch.Tensor:
        if width <= 0 or height <= 0:
            return tensor
        # Convert to NCHW float32 for interpolation.
        source_dtype = tensor.dtype
        needs_cast = source_dtype != torch.float32
        if needs_cast:
            work = tensor.to(dtype=torch.float32)
        else:
            work = tensor
        work = work.permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            work,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        resized = resized.squeeze(0).permute(1, 2, 0)
        if needs_cast:
            if source_dtype.is_floating_point:
                resized = resized.to(dtype=source_dtype)
            else:
                resized = torch.clamp(resized, 0, 255).to(dtype=source_dtype)
        return resized.contiguous()

    def _ensure_tensor(self, frame: torch.Tensor) -> torch.Tensor:
        if isinstance(frame, torch.Tensor):
            if frame.dim() != 3:
                raise ValueError("Expected frame tensor with shape [H,W,C]")
            return frame
        npFrame = np.asarray(frame)
        if npFrame.ndim != 3:
            raise ValueError("Expected frame array with shape [H,W,C]")
        tensor = torch.from_numpy(npFrame)
        return tensor


class VideoPipelineManager:
    """Manage frame ingest, viewport composition and stats."""

    def __init__(
        self,
        width: int,
        height: int,
        stats: StatsCollector,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._ingest = FrameIngest(depth=2)
        self._composer = ViewportComposer(width, height)
        self._stats = stats
        self._loop = loop
        self._width = width
        self._height = height

    async def submit_frame(self, frame: torch.Tensor) -> FrameRecord:
        tensor = self._normalize_frame(frame)
        record = await self._ingest.push(tensor)
        await self._stats.record_frame_ingest(
            queue_depth=self._ingest.depth, dropped=self._ingest.dropped
        )
        return record

    async def next_frame(self, after_sequence: int) -> FrameRecord:
        return await self._ingest.wait_for_next(after_sequence)

    async def compose_view(
        self, record: FrameRecord, viewport: ViewportState
    ) -> torch.Tensor:
        return self._composer.compose(record.tensor, viewport)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _normalize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if not isinstance(frame, torch.Tensor):
            frame = torch.as_tensor(frame)
        if frame.dim() == 2:
            frame = frame.unsqueeze(-1)
        if frame.dim() != 3:
            raise ValueError("Frame must have shape [H,W,C]")
        if frame.shape[2] not in (3, 4):
            raise ValueError("Frame must have 3 (RGB) or 4 (RGBA) channels")
        if not frame.is_contiguous():
            frame = frame.contiguous()
        return frame
