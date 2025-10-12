from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import replace
from typing import Callable, Dict, Optional, Tuple

from aiohttp import web
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    RTCIceServer,
    RTCRtpSender,
    MediaStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from aiortc.rtp import RtcpPsfbPacket, RTCP_PSFB_PLI

from .config import StreamCaps, WindowConfig
from .encoder import EncoderUnavailable, LosslessEncoder, SoftwareLosslessEncoder, create_encoder
from .pipeline import VideoPipelineManager, ViewportState
from .stats import StatsCollector

logger = logging.getLogger(__name__)

# aiortc does not expose the FIR fmt constant, define it locally (fmt = 4)
RTCP_PSFB_FIR = 4


class LosslessTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        client_id: str,
        pipeline: VideoPipelineManager,
        viewport: ViewportState,
        encoder: LosslessEncoder,
        stats: StatsCollector,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        super().__init__()
        self._client_id = client_id
        self._pipeline = pipeline
        self._viewport = viewport
        self._encoder = encoder
        self._stats = stats
        self._loop = loop
        self._last_sequence = 0
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._last_sent_at: Optional[float] = None
        self._frames_dropped = 0
        self._encoder_profile = getattr(encoder, "profile_level_id", "42e01f")
        self._encoder_sprop = getattr(encoder, "sprop_parameter_sets", None)
        self._idr_cooldown_seconds = 0.5
        self._last_forced_idr = 0.0
        self._forced_idr_count = 0
        self._ingest_drop_limit = max(3, int(max(self._viewport.fps_cap, 30) * 0.1))

    @property
    def viewport(self) -> ViewportState:
        return self._viewport

    @property
    def encoder_mode(self) -> str:
        return "hardware" if self._encoder.hardware else "software"

    @property
    def encoder_detail(self) -> str:
        return self._encoder.detail

    @property
    def encoder_profile(self) -> str:
        return self._encoder_profile

    @property
    def encoder_sprop(self) -> Optional[str]:
        return getattr(self._encoder, "sprop_parameter_sets", self._encoder_sprop)

    def update_viewport(self, viewport: ViewportState) -> None:
        self._viewport = viewport
        self._ingest_drop_limit = max(3, int(max(viewport.fps_cap, 30) * 0.1))

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self._pause_event.clear()
        else:
            self._pause_event.set()

    async def recv(self):
        await self._pause_event.wait()
        record = await self._pipeline.next_frame(self._last_sequence)
        dropped = max(0, record.sequence - self._last_sequence - 1)
        if dropped:
            self._frames_dropped += dropped
            await self._stats.record_dropped(self._client_id, self._frames_dropped)
            if dropped <= self._ingest_drop_limit:
                await self._maybe_force_idr("ingest_drop", drop_count=dropped)
            else:
                logger.debug(
                    "Ingest overshoot detected (drop=%d threshold=%d, client=%s) – skipping forced IDR",
                    dropped,
                    self._ingest_drop_limit,
                    self._client_id,
                )

        # FPS throttling.
        if self._viewport.fps_cap > 0:
            interval = 1.0 / self._viewport.fps_cap
            now = time.perf_counter()
            if self._last_sent_at is not None and now - self._last_sent_at < interval:
                await asyncio.sleep(interval - (now - self._last_sent_at))
            self._last_sent_at = time.perf_counter()

        composed = await self._pipeline.compose_view(record, self._viewport)
        composed = composed.contiguous()

        async def _encode() -> object:
            return self._encoder.encode(composed, self._viewport)

        start = time.perf_counter()
        if isinstance(self._encoder, SoftwareLosslessEncoder):
            packet = await _encode()
        else:
            packet = await self._loop.run_in_executor(None, self._encoder.encode, composed, self._viewport)
        duration_ms = (time.perf_counter() - start) * 1000.0
        await self._stats.record_encode(self._client_id, duration_ms)
        self._last_sequence = record.sequence
        return packet

    async def stop(self) -> None:
        super().stop()
        if hasattr(self._encoder, "finish"):
            self._encoder.finish()

    async def handle_rtcp_feedback(self, feedback: str) -> None:
        await self._maybe_force_idr(feedback)

    async def _maybe_force_idr(self, reason: str, drop_count: int = 0) -> None:
        if not hasattr(self._encoder, "force_idr"):
            return

        now = time.perf_counter()
        elapsed = now - self._last_forced_idr
        if elapsed < self._idr_cooldown_seconds:
            logger.debug(
                "IDR cooldown active (reason=%s, dt=%.3fs, client=%s)",
                reason,
                elapsed,
                self._client_id,
            )
            return

        try:
            self._encoder.force_idr()
        except Exception:  # pragma: no cover - best effort
            logger.warning(
                "Failed to force IDR (reason=%s, client=%s)",
                reason,
                self._client_id,
                exc_info=True,
            )
            return

        self._last_forced_idr = now
        self._forced_idr_count += 1
        logger.info(
            "Forced IDR (reason=%s, drop=%d, total=%d, client=%s)",
            reason,
            drop_count,
            self._forced_idr_count,
            self._client_id,
        )
        await self._stats.record_forced_idr(self._client_id, reason)


async def maybe_await(result):
    if asyncio.iscoroutine(result):
        return await result
    return result


class SubscriberSession:
    def __init__(
        self,
        client_id: str,
        pc: RTCPeerConnection,
        track: Optional[LosslessTrack],
        viewport: ViewportState,
        stats: StatsCollector,
        caps: StreamCaps,
        pipeline_backed: bool = True,
        forward: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.client_id = client_id
        self.pc = pc
        self.track = track
        self.stats = stats
        self.caps = caps
        self._paused = False
        self._viewport = viewport
        self._pipeline_backed = pipeline_backed
        self._forward = forward
        self._sender: Optional[RTCRtpSender] = None

    @property
    def viewport(self) -> ViewportState:
        return self._viewport

    async def apply_command(self, message: str) -> None:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON payload from %s: %s", self.client_id, message)
            return

        if not self._pipeline_backed:
            payload.setdefault("client", self.client_id)
            if self._forward:
                await maybe_await(self._forward(json.dumps(payload)))
            return

        command = payload.get("cmd")
        if command == "pause":
            self._paused = True
            if self.track:
                self.track.set_paused(True)
        elif command == "resume":
            self._paused = False
            if self.track:
                self.track.set_paused(False)
        elif command == "zoom":
            center = payload.get("center", self._viewport.center_px)
            scale = float(payload.get("scale", self._viewport.zoom))
            viewport = replace(
                self._viewport,
                center_px=(int(center[0]), int(center[1])),
                zoom=max(1.0, float(scale)),
            )
            self._viewport = viewport
            if self.track:
                self.track.update_viewport(viewport)
        elif command == "pan":
            dx = int(payload.get("dx", 0))
            dy = int(payload.get("dy", 0))
            cx, cy = self._viewport.center_px
            viewport = replace(
                self._viewport,
                center_px=(cx + dx, cy + dy),
            )
            self._viewport = viewport
            if self.track:
                self.track.update_viewport(viewport)
        elif command == "resize":
            width = int(payload.get("w", self._viewport.target_size[0]))
            height = int(payload.get("h", self._viewport.target_size[1]))
            fps = float(payload.get("fps", self._viewport.fps_cap))
            clamped_w, clamped_h, clamped_fps = self.caps.clamp(width, height, int(fps))
            viewport = replace(
                self._viewport,
                target_size=(clamped_w, clamped_h),
                fps_cap=clamped_fps,
            )
            self._viewport = viewport
            if self.track:
                self.track.update_viewport(viewport)
            await self.stats.update_viewport(
                self.client_id, clamped_w, clamped_h, clamped_fps
            )
        elif command == "select_view":
            view_id = payload.get("id", self._viewport.view_id)
            viewport = replace(self._viewport, view_id=view_id)
            self._viewport = viewport
            if self.track:
                self.track.update_viewport(viewport)
        elif command == "metrics":
            if "rtt" in payload:
                await self.stats.record_rtt(self.client_id, float(payload["rtt"]))
        else:
            logger.info("Unhandled control message %s from %s", payload.get("cmd"), self.client_id)

    async def close(self) -> None:
        await self.pc.close()
        await self.stats.unregister_client(self.client_id)
        if self.track:
            await self.track.stop()

    def attach_sender(self, sender: Optional[RTCRtpSender]) -> None:
        self._sender = sender
        if sender is None:
            return
        original_handler = sender._handle_rtcp_packet

        async def _intercept(packet):
            if isinstance(packet, RtcpPsfbPacket):
                if packet.fmt == RTCP_PSFB_PLI and self.track:
                    await self.track.handle_rtcp_feedback("rtcp_pli")
                elif packet.fmt == RTCP_PSFB_FIR and self.track:
                    await self.track.handle_rtcp_feedback("rtcp_fir")
            await original_handler(packet)

        sender._handle_rtcp_packet = _intercept


class EmbeddedSFU:
    """Pure Python SFU built on aiortc for lossless torchwindow streaming."""

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
        self._subscribers: Dict[str, SubscriberSession] = {}
        self._app = web.Application()
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._caps = config.stream_caps
        self._actual_port: Optional[int] = None
        self._relay = MediaRelay()
        self._remote_pc: Optional[RTCPeerConnection] = None
        self._remote_track: Optional[MediaStreamTrack] = None
        self._publisher_channel = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        self._app.router.add_get("/", self._handle_index)
        self._app.router.add_get("/app.js", self._handle_static_js)
        self._app.router.add_get("/app.css", self._handle_static_css)
        self._app.router.add_get("/metrics.json", self._handle_metrics)
        self._app.router.add_post("/whep", self._handle_whep)
        self._app.router.add_post("/whip", self._handle_whip)

    async def start(self) -> None:
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(
            self._runner, host=self._config.http_host, port=self._config.http_port
        )
        await self._site.start()
        if self._site._server and self._site._server.sockets:
            sock = self._site._server.sockets[0]
            self._actual_port = sock.getsockname()[1]
        logger.info(
            "TorchWindow embedded SFU listening on http://%s:%s",
            self._config.http_host,
            self.server_port,
        )

    async def stop(self) -> None:
        for session in list(self._subscribers.values()):
            await session.close()
        self._subscribers.clear()
        if self._site:
            await self._site.stop()
            self._site = None
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._actual_port = None

    async def _handle_index(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(self._static_path("index.html"))

    async def _handle_static_js(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(self._static_path("app.js"))

    async def _handle_static_css(self, request: web.Request) -> web.StreamResponse:
        return web.FileResponse(self._static_path("app.css"))

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        snapshot = await self._stats.snapshot(self._config.name)
        return web.json_response(snapshot)

    async def _handle_whep(self, request: web.Request) -> web.StreamResponse:
        offer_sdp = await request.text()
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        pc = self._create_peer_connection()
        client_id = str(uuid.uuid4())

        pipeline_backed = self._remote_track is None
        offer_lower = offer_sdp.lower()
        high444_requested = False  # Disable 4:4:4 until explicit support is negotiated
        pixel_format = "NV12"
        viewport = ViewportState(
            view_id="default",
            center_px=(self._pipeline.width // 2, self._pipeline.height // 2),
            zoom=1.0,
            target_size=(self._caps.max_width, self._caps.max_height),
            fps_cap=self._caps.max_fps,
        )

        if pipeline_backed:
            prefer_hw = self._config.prefer_hardware_encode
            try:
                encoder = create_encoder(
                    width=viewport.target_size[0],
                    height=viewport.target_size[1],
                    codec=self._config.codec,
                    fps=float(viewport.fps_cap),
                    prefer_hardware=prefer_hw,
                    pixel_format=pixel_format,
                )
            except EncoderUnavailable as exc:
                await pc.close()
                return web.Response(status=500, text=str(exc))
            if getattr(encoder, "profile_level_id", None) != "f4001f":
                logger.warning(
                    "Streaming with chroma-sub-sampled H.264 for client compatibility (profile-level-id=%s).",
                    getattr(encoder, "profile_level_id", "unknown"),
                )

            track = LosslessTrack(
                client_id=client_id,
                pipeline=self._pipeline,
                viewport=viewport,
                encoder=encoder,
                stats=self._stats,
                loop=self._loop,
            )
            session = SubscriberSession(
                client_id,
                pc,
                track,
                viewport,
                self._stats,
                self._caps,
            )
            sender = pc.addTrack(track)
            session.attach_sender(sender)
            await self._stats.set_encoder_mode(
                track.encoder_mode,
                track.encoder_detail,
            )
        else:
            if not self._remote_track:
                await pc.close()
                return web.Response(status=503, text="Publisher not connected")
            relay_track = self._relay.subscribe(self._remote_track)
            session = SubscriberSession(
                client_id,
                pc,
                None,
                viewport,
                self._stats,
                self._caps,
                pipeline_backed=False,
                forward=self._forward_command,
            )
            sender = pc.addTrack(relay_track)
            session.attach_sender(sender)
            await self._stats.set_encoder_mode("remote", "relay")

        await self._stats.register_client(
            client_id,
            viewport.target_size[0],
            viewport.target_size[1],
            viewport.fps_cap,
        )
        self._register_session(session)

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            if pc.connectionState in {"failed", "closed"}:
                await self._dispose_session(client_id)

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            logger.debug("Datachannel opened for %s: %s", client_id, channel.label)

            @channel.on("message")
            async def on_message(message) -> None:
                await session.apply_command(message)

        await pc.setRemoteDescription(offer)
        _prioritize_h264_codecs(pc, offer.sdp)
        logger.warning("Sender codecs after prioritization: %s", _describe_transceiver_codecs(pc))
        answer = await pc.createAnswer()
        base_sdp = answer.sdp
        logger.warning("Local SDP before patch:\n%s", base_sdp)
        track_profile = getattr(session.track, "encoder_profile", None) if hasattr(session, "track") else None
        track_sprop = getattr(session.track, "encoder_sprop", None) if hasattr(session, "track") else None
        if track_profile is None and pipeline_backed:
            track_profile = getattr(track, "encoder_profile", None)
        if track_sprop is None and pipeline_backed:
            track_sprop = getattr(track, "encoder_sprop", None)
        profile_hex = track_profile or "42e01f"
        h264_payloads, keep_payloads = _discover_h264_payloads(base_sdp)
        if not h264_payloads:
            logger.warning("No H.264 payloads found in local SDP; returning unmodified answer.")
            patched = base_sdp
        else:
            patched = _patch_h264_profile(
                base_sdp,
                profile=profile_hex,
                sprop=track_sprop,
                payloads=h264_payloads,
            )
            patched = _filter_h264_only(patched, payloads=keep_payloads)
        patched_answer = RTCSessionDescription(sdp=patched, type=answer.type)
        await pc.setLocalDescription(patched_answer)
        await _await_ice_gathering_complete(pc)
        final_sdp = pc.localDescription.sdp
        logger.warning("Local SDP after gather:\n%s", final_sdp)
        logger.info("Subscriber %s connected", client_id)
        return web.Response(
            status=200,
            content_type="application/sdp",
            text=final_sdp,
        )

    async def _handle_whip(self, request: web.Request) -> web.StreamResponse:
        offer_sdp = await request.text()
        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")

        if self._remote_pc:
            await self._teardown_remote()

        pc = self._create_peer_connection()
        self._remote_pc = pc

        @pc.on("track")
        def on_track(track: MediaStreamTrack) -> None:
            if track.kind == "video":
                self._remote_track = track
                logger.info("Remote publisher track received (%s)", track.kind)

        @pc.on("datachannel")
        def on_channel(channel) -> None:
            self._publisher_channel = channel
            logger.info("Publisher control channel: %s", channel.label)

        @pc.on("connectionstatechange")
        async def on_state_change() -> None:
            if pc.connectionState in {"failed", "closed"}:
                await self._teardown_remote()

        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return web.Response(
            status=200,
            content_type="application/sdp",
            text=pc.localDescription.sdp,
        )

    def _static_path(self, filename: str) -> str:
        import importlib.resources

        return str(importlib.resources.files("torchwindow.static").joinpath(filename))

    def _create_peer_connection(self) -> RTCPeerConnection:
        ice_servers = []
        if self._config.stun_servers:
            ice_servers.append(RTCIceServer(urls=list(self._config.stun_servers)))
        if self._config.turn_servers:
            ice_servers.append(RTCIceServer(urls=list(self._config.turn_servers)))
        configuration = RTCConfiguration(iceServers=ice_servers)
        return RTCPeerConnection(configuration=configuration)

    def _register_session(self, session: SubscriberSession) -> None:
        self._subscribers[session.client_id] = session

    async def _dispose_session(self, client_id: str) -> None:
        session = self._subscribers.pop(client_id, None)
        if session:
            await session.close()
            logger.info("Subscriber %s disconnected", client_id)

    @property
    def server_port(self) -> int:
        if self._actual_port is not None:
            return self._actual_port
        return self._config.http_port

    async def update_caps(self, caps: StreamCaps) -> None:
        self._caps = caps
        for session in self._subscribers.values():
            viewport = session.viewport
            new_w, new_h, new_fps = caps.clamp(
                viewport.target_size[0], viewport.target_size[1], viewport.fps_cap
            )
            new_viewport = replace(
                viewport,
                target_size=(new_w, new_h),
                fps_cap=new_fps,
            )
            if session.track:
                session.track.update_viewport(new_viewport)
            await self._stats.update_viewport(
                session.client_id, new_w, new_h, new_fps
            )

    async def _teardown_remote(self) -> None:
        if self._remote_pc:
            await self._remote_pc.close()
        self._remote_pc = None
        self._remote_track = None
        self._publisher_channel = None

    async def _forward_command(self, message: str) -> None:
        channel = self._publisher_channel
        if channel and channel.readyState == "open":
            channel.send(message)


def _prioritize_h264_codecs(pc: RTCPeerConnection, offer_sdp: str) -> None:
    offer_h264_payloads, _ = _discover_h264_payloads(offer_sdp)
    offer_payload_ints = set()
    for pt in offer_h264_payloads:
        try:
            offer_payload_ints.add(int(pt))
        except (TypeError, ValueError):
            continue

    for transceiver in pc.getTransceivers():
        codecs = list(getattr(transceiver, "_codecs", []))
        if not codecs:
            continue
        h264_codecs = [
            codec
            for codec in codecs
            if getattr(codec, "mimeType", "").lower() == "video/h264"
        ]
        if not h264_codecs:
            continue
        preferred = None
        for codec in h264_codecs:
            pt = getattr(codec, "payloadType", None)
            try:
                pt_int = int(pt)
            except (TypeError, ValueError):
                pt_int = None
            if pt_int in offer_payload_ints:
                preferred = codec
                break
        if preferred is None:
            preferred = h264_codecs[0]
        if codecs[0] is not preferred:
            reordered = [preferred] + [codec for codec in codecs if codec is not preferred]
            setattr(transceiver, "_codecs", reordered)
            logger.warning(
                "Prioritized H.264 payloadType=%s for mid=%s",
                getattr(preferred, "payloadType", None),
                transceiver.mid,
            )


def _describe_transceiver_codecs(pc: RTCPeerConnection) -> dict[str, list[tuple]]:
    summary: dict[str, list[tuple]] = {}
    for index, transceiver in enumerate(pc.getTransceivers()):
        key = transceiver.mid or f"{transceiver.kind}:{index}"
        codecs = getattr(transceiver, "_codecs", [])
        summary[key] = [
            (getattr(codec, "payloadType", None), getattr(codec, "mimeType", None))
            for codec in codecs
        ]
    return summary


def _patch_h264_profile(
    sdp: str, profile: str, sprop: Optional[str] = None, payloads: Optional[set[str]] = None
) -> str:
    target_payloads = payloads or set()
    lines: list[str] = []
    for line in sdp.splitlines():
        if line.startswith("a=fmtp:"):
            try:
                prefix, params = line.split(" ", 1)
            except ValueError:
                lines.append(line)
                continue
            payload = prefix.split(":", 1)[1]
            if target_payloads and payload not in target_payloads:
                lines.append(line)
                continue
            items = [item.strip() for item in params.split(";") if item.strip()]
            updated: list[str] = []
            has_profile = False
            has_sprop = False
            for item in items:
                if item.startswith("profile-level-id="):
                    updated.append(f"profile-level-id={profile}")
                    has_profile = True
                elif item.startswith("sprop-parameter-sets="):
                    if sprop:
                        updated.append(f"sprop-parameter-sets={sprop}")
                    has_sprop = True
                else:
                    updated.append(item)
            if not has_profile:
                updated.append(f"profile-level-id={profile}")
            if sprop and not has_sprop:
                updated.append(f"sprop-parameter-sets={sprop}")
            line = f"a=fmtp:{payload} {';'.join(updated)}"
        lines.append(line)
    return "\r\n".join(lines) + "\r\n"


def _filter_h264_only(sdp: str, payloads: set[str]) -> str:
    lines = sdp.splitlines()
    filtered: list[str] = []
    current_media = None

    def keep_payload_line(line: str) -> bool:
        if ":" not in line:
            return True
        prefix, rest = line.split(":", 1)
        if prefix not in {"a=rtpmap", "a=rtcp-fb", "a=fmtp"}:
            return True
        payload = rest.split()[0]
        return payload in payloads

    for line in lines:
        if line.startswith("m=video"):
            parts = line.split()
            header = parts[:3]
            pts = [pt for pt in parts[3:] if pt in payloads]
            filtered.append(" ".join(header + pts))
            current_media = "video"
            continue
        if line.startswith("m="):
            current_media = None
            filtered.append(line)
            continue
        if current_media == "video":
            if keep_payload_line(line):
                filtered.append(line)
        else:
            filtered.append(line)
    return "\r\n".join(filtered) + "\r\n"


def _discover_h264_payloads(sdp: str) -> Tuple[set[str], set[str]]:
    """
    Inspect aiortc-generated SDP and return the dynamic payload types for the H.264
    codec and any associated RTX entries. We keep RTX payloads so retransmissions
    continue to work after we trim the m=video line.
    """
    h264_payloads: set[str] = set()
    for line in sdp.splitlines():
        if not line.startswith("a=rtpmap:"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        payload = parts[0].split(":", 1)[1]
        encoding = parts[1].split()[0]
        if encoding.upper().startswith("H264/"):
            h264_payloads.add(payload)

    keep_payloads = set(h264_payloads)
    if not h264_payloads:
        return h264_payloads, keep_payloads

    for line in sdp.splitlines():
        if not line.startswith("a=fmtp:"):
            continue
        try:
            prefix, params = line.split(" ", 1)
        except ValueError:
            continue
        payload = prefix.split(":", 1)[1]
        for item in params.split(";"):
            item = item.strip()
            if item.startswith("apt="):
                apt = item.split("=", 1)[1]
                if apt in h264_payloads:
                    keep_payloads.add(payload)
    return h264_payloads, keep_payloads


async def _await_ice_gathering_complete(pc: RTCPeerConnection, timeout: float = 5.0) -> None:
    """
    Ensure ICE gathering reaches the 'complete' state before we snapshot the local SDP.
    This avoids returning an answer with port 9 / 0.0.0.0 when the client does not support trickle ICE.
    """
    if pc.iceGatheringState == "complete":
        return
    deadline = asyncio.get_event_loop().time() + timeout
    while pc.iceGatheringState != "complete" and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
    if pc.iceGatheringState != "complete":
        logger.warning("ICE gathering did not complete within %.1fs", timeout)
