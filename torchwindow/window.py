from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional, Sequence

import aiohttp
from aiortc import RTCConfiguration, RTCPeerConnection, RTCIceServer

from .config import StreamCaps, WindowConfig
from .encoder import EncoderUnavailable, create_encoder
from .pipeline import VideoPipelineManager, ViewportState
from .sfu import EmbeddedSFU, LosslessTrack
from .stats import StatsCollector

logger = logging.getLogger(__name__)


class RemotePublisher:
    def __init__(
        self,
        config: WindowConfig,
        pipeline: VideoPipelineManager,
        stats: StatsCollector,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._config = config
        self._pipeline = pipeline
        self._stats = stats
        self._loop = loop
        self._pc: Optional[RTCPeerConnection] = None
        self._channel = None
        self._track: Optional[LosslessTrack] = None
        self._viewport = ViewportState(
            view_id="default",
            center_px=(pipeline.width // 2, pipeline.height // 2),
            zoom=1.0,
            target_size=(config.max_viewport_width, config.max_viewport_height),
            fps_cap=config.max_viewport_fps,
        )
        self._caps = config.stream_caps

    def _rtc_configuration(self) -> RTCConfiguration:
        ice_servers = []
        if self._config.stun_servers:
            ice_servers.append(RTCIceServer(urls=list(self._config.stun_servers)))
        if self._config.turn_servers:
            ice_servers.append(RTCIceServer(urls=list(self._config.turn_servers)))
        return RTCConfiguration(iceServers=ice_servers)

    async def start(self) -> None:
        if self._pc:
            return
        self._pc = RTCPeerConnection(configuration=self._rtc_configuration())
        channel = self._pc.createDataChannel("control")
        self._channel = channel

        encoder = create_encoder(
            width=self._viewport.target_size[0],
            height=self._viewport.target_size[1],
            codec=self._config.codec,
            fps=float(self._viewport.fps_cap),
            prefer_hardware=self._config.prefer_hardware_encode,
        )
        self._track = LosslessTrack(
            client_id="publisher",
            pipeline=self._pipeline,
            viewport=self._viewport,
            encoder=encoder,
            stats=self._stats,
            loop=self._loop,
        )
        await self._stats.register_client(
            "publisher",
            self._viewport.target_size[0],
            self._viewport.target_size[1],
            self._viewport.fps_cap,
        )
        self._pc.addTrack(self._track)
        await self._stats.set_encoder_mode(
            "hardware" if self._track.encoder_mode == "hardware" else "software",
            self._track.encoder_detail,
        )

        @channel.on("message")
        async def on_message(message) -> None:
            await self._handle_command(message)

        @self._pc.on("connectionstatechange")
        async def on_state_change() -> None:
            if self._pc and self._pc.connectionState in {"failed", "closed"}:
                await self.stop()

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        endpoint = self._config.sfu_url.rstrip("/") + "/whip"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                data=offer.sdp,
                headers={"Content-Type": "application/sdp"},
            ) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"WHIP publish failed: {resp.status}")
                answer_sdp = await resp.text()

        await self._pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type="answer")
        )

    async def stop(self) -> None:
        if self._pc:
            await self._pc.close()
        self._pc = None
        if self._track:
            await self._track.stop()
        self._track = None
        await self._stats.unregister_client("publisher")

    async def update_caps(self, caps: StreamCaps) -> None:
        self._caps = caps
        width, height, fps = caps.clamp(
            self._viewport.target_size[0],
            self._viewport.target_size[1],
            self._viewport.fps_cap,
        )
        self._viewport = ViewportState(
            view_id=self._viewport.view_id,
            center_px=self._viewport.center_px,
            zoom=self._viewport.zoom,
            target_size=(width, height),
            fps_cap=fps,
        )
        if self._track:
            self._track.update_viewport(self._viewport)
            await self._stats.update_viewport("publisher", width, height, fps)

    async def _handle_command(self, message: str) -> None:
        if not self._track:
            return
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Invalid control message from SFU: %s", message)
            return
        cmd = payload.get("cmd")
        if cmd == "pause":
            self._track.set_paused(True)
        elif cmd == "resume":
            self._track.set_paused(False)
        elif cmd == "zoom":
            center = payload.get("center", self._viewport.center_px)
            scale = float(payload.get("scale", self._viewport.zoom))
            self._viewport = ViewportState(
                view_id=self._viewport.view_id,
                center_px=(int(center[0]), int(center[1])),
                zoom=max(1.0, scale),
                target_size=self._viewport.target_size,
                fps_cap=self._viewport.fps_cap,
            )
            self._track.update_viewport(self._viewport)
        elif cmd == "pan":
            dx = int(payload.get("dx", 0))
            dy = int(payload.get("dy", 0))
            cx, cy = self._viewport.center_px
            self._viewport = ViewportState(
                view_id=self._viewport.view_id,
                center_px=(cx + dx, cy + dy),
                zoom=self._viewport.zoom,
                target_size=self._viewport.target_size,
                fps_cap=self._viewport.fps_cap,
            )
            self._track.update_viewport(self._viewport)
        elif cmd == "resize":
            width = int(payload.get("w", self._viewport.target_size[0]))
            height = int(payload.get("h", self._viewport.target_size[1]))
            fps = float(payload.get("fps", self._viewport.fps_cap))
            width, height, fps = self._caps.clamp(width, height, int(fps))
            self._viewport = ViewportState(
                view_id=self._viewport.view_id,
                center_px=self._viewport.center_px,
                zoom=self._viewport.zoom,
                target_size=(width, height),
                fps_cap=fps,
            )
            self._track.update_viewport(self._viewport)
            await self._stats.update_viewport("publisher", width, height, fps)
        elif cmd == "metrics" and "rtt" in payload:
            await self._stats.record_rtt("publisher", float(payload["rtt"]))
class Window:
    """Main entry point for TorchWindow 2.0 lossless streaming."""

    def __init__(self, **kwargs) -> None:
        self._config = WindowConfig(**kwargs)
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="torchwindow-loop", daemon=True
        )
        self._loop_thread.start()

        self._running = True
        self._stats: Optional[StatsCollector] = None
        self._pipeline: Optional[VideoPipelineManager] = None
        self._sfu: Optional[EmbeddedSFU] = None
        self._publisher: Optional[RemotePublisher] = None

        fut = asyncio.run_coroutine_threadsafe(self._async_init(), self._loop)
        fut.result()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def config(self) -> WindowConfig:
        return self._config

    @property
    def encoder_mode(self) -> str:
        if self._stats:
            return self._stats.encoder_mode
        return "unknown"

    @property
    def encoder_detail(self) -> Optional[str]:
        if self._stats:
            return self._stats.encoder_detail
        return None

    @property
    def http_port(self) -> int:
        if self._sfu:
            return self._sfu.server_port
        return self._config.http_port

    def set_stream_caps(self, max_width: int, max_height: int, max_fps: int) -> None:
        self._config.max_viewport_width = max_width
        self._config.max_viewport_height = max_height
        self._config.max_viewport_fps = max_fps
        if self._sfu:
            caps = StreamCaps(max_width, max_height, max_fps)
            sfu = self._sfu

            def _dispatch() -> None:
                asyncio.create_task(sfu.update_caps(caps))

            self._loop.call_soon_threadsafe(_dispatch)
        if self._publisher:
            caps = StreamCaps(max_width, max_height, max_fps)

            def _dispatch_pub() -> None:
                asyncio.create_task(self._publisher.update_caps(caps))

            self._loop.call_soon_threadsafe(_dispatch_pub)

    def draw(self, frame) -> None:
        if not self._running or not self._pipeline:
            return
        fut = asyncio.run_coroutine_threadsafe(
            self._pipeline.submit_frame(frame), self._loop
        )

        def _log_result(f):
            try:
                f.result()
            except Exception:  # pragma: no cover - best effort logging
                logger.exception("Failed to submit frame to pipeline")

        fut.add_done_callback(_log_result)

    def close(self) -> None:
        if not self._running:
            return
        fut = asyncio.run_coroutine_threadsafe(self._async_shutdown(), self._loop)
        fut.result()
        self._running = False
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=2.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _async_init(self) -> None:
        self._stats = StatsCollector()
        self._pipeline = VideoPipelineManager(
            self._config.width, self._config.height, self._stats, self._loop
        )
        if self._config.stream and self._config.sfu_mode == "embedded":
            self._sfu = EmbeddedSFU(self._config, self._pipeline, self._stats, self._loop)
            await self._sfu.start()
        elif self._config.stream and self._config.sfu_mode == "remote":
            self._publisher = RemotePublisher(
                self._config, self._pipeline, self._stats, self._loop
            )
            try:
                await self._publisher.start()
            except EncoderUnavailable:
                logger.exception("NVENC encoder unavailable for remote publishing")
                raise

    async def _async_shutdown(self) -> None:
        if self._sfu:
            await self._sfu.stop()
            self._sfu = None
        if self._publisher:
            await self._publisher.stop()
            self._publisher = None
