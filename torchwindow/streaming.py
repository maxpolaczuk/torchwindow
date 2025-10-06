import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import VideoStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from av import VideoFrame

import numpy as np

logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "static"


class FrameStreamTrack(VideoStreamTrack):
    """WebRTC video track backed by an asyncio queue of frames."""

    def __init__(self, frame_queue: "asyncio.Queue"):
        super().__init__()
        self._queue = frame_queue
        self._start = asyncio.get_event_loop().time()

    async def recv(self):
        frame = await self._queue.get()
        video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


@dataclass
class StreamConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    index_file: Optional[Path] = None
    queue_size: int = 3
    max_bitrate: Optional[int] = None
    min_bitrate: Optional[int] = None
    max_framerate: Optional[float] = None
    codec: Optional[str] = None


class WebRTCStreamer:
    def __init__(self, config: StreamConfig, name: str):
        self.config = config
        self.name = name
        self._frame_queue: "asyncio.Queue" = asyncio.Queue(
            maxsize=config.queue_size
        )
        self._pcs: Set[RTCPeerConnection] = set()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._started = threading.Event()
        self._closed = False

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def start(self):
        if self._thread.is_alive():
            return
        self._thread.start()
        future = asyncio.run_coroutine_threadsafe(self._async_start(), self._loop)
        future.result()
        self._started.set()
        logger.info(
            "WebRTC streamer available at http://%s:%d",
            self.config.host,
            self.config.port,
        )

    async def _async_start(self):
        self._app = web.Application()
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_post("/offer", self._handle_offer)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.config.host, self.config.port)
        await self._site.start()

    async def _handle_index(self, request):
        index_path = self.config.index_file or ASSETS_DIR / "index.html"
        return web.FileResponse(index_path)

    async def _handle_offer(self, request):
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc = RTCPeerConnection()
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await pc.close()
                self._pcs.discard(pc)

        track = FrameStreamTrack(self._frame_queue)
        sender = pc.addTrack(track)

        transceiver = next((t for t in pc.getTransceivers() if t.sender == sender), None)
        if (
            transceiver
            and self.config.codec
            and hasattr(transceiver, "setCodecPreferences")
        ):
            codec_name = self.config.codec.lower()
            capabilities = RTCRtpSender.getCapabilities("video").codecs
            preferred = [c for c in capabilities if c.mimeType.lower() == f"video/{codec_name}"]
            if preferred:
                try:
                    transceiver.setCodecPreferences(preferred)
                except Exception as exc:  # pragma: no cover - codec preference best effort
                    logger.warning("Unable to set codec preferences: %s", exc)

        params = None
        if hasattr(sender, "getParameters"):
            params = sender.getParameters()
        elif hasattr(sender, "get_parameters"):
            params = sender.get_parameters()

        if params and getattr(params, "encodings", None):
            encoding = params.encodings[0]
            if self.config.max_bitrate:
                encoding.maxBitrate = self.config.max_bitrate
            if self.config.min_bitrate:
                encoding.minBitrate = self.config.min_bitrate
            if self.config.max_framerate:
                encoding.maxFramerate = self.config.max_framerate
            setter = None
            if hasattr(sender, "setParameters"):
                setter = sender.setParameters
            elif hasattr(sender, "set_parameters"):
                setter = sender.set_parameters
            if setter:
                try:
                    await setter(params)
                except Exception as exc:  # pragma: no cover - encoder negotiation best effort
                    logger.warning("Unable to apply sender parameters: %s", exc)

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.json_response(
            {
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type,
            }
        )

    def send(self, frame: np.ndarray):
        if not self._thread.is_alive():
            raise RuntimeError("streamer not started")
        frame = np.ascontiguousarray(frame)
        asyncio.run_coroutine_threadsafe(self._enqueue_frame(frame), self._loop)

    async def _enqueue_frame(self, frame: np.ndarray):
        while self._frame_queue.qsize() >= self.config.queue_size:
            try:
                self._frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        await self._frame_queue.put(frame)

    def close(self):
        if not self._thread.is_alive():
            if not self._closed:
                asyncio.run_coroutine_threadsafe(self.close_async(), self._loop).result()
            return
        asyncio.run_coroutine_threadsafe(self.close_async(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=1)
        self._loop.close()

    async def close_async(self):
        if self._closed:
            return
        self._closed = True
        for pc in list(self._pcs):
            await pc.close()
        self._pcs.clear()
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        if not self._frame_queue.empty():
            self._frame_queue = asyncio.Queue(maxsize=self.config.queue_size)


__all__ = ["StreamConfig", "WebRTCStreamer"]
