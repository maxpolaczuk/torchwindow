from __future__ import annotations

import base64
import importlib
import inspect
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Tuple, Union

import av
import torch
from aiortc.codecs.h264 import VIDEO_TIME_BASE
from av.video.frame import VideoFrame

from .pipeline import ViewportState

logger = logging.getLogger(__name__)


class EncoderUnavailable(RuntimeError):
    pass


def _load_pynvcodec():
    try:
        return importlib.import_module("PyNvVideoCodec")
    except ModuleNotFoundError as exc:
        raise EncoderUnavailable(
            "PyNvVideoCodec is not installed. Install torchwindow[nvenc] to enable NVENC."
        ) from exc


def _ticks_per_frame(fps: float) -> int:
    if fps <= 0:
        return 0
    return max(1, int(round(90000.0 / fps)))


def _ensure_rgb_tensor(frame: torch.Tensor) -> torch.Tensor:
    if frame.shape[2] == 3:
        return frame
    if frame.shape[2] == 4:
        return frame[..., :3]
    raise ValueError("Frame must have 3 (RGB) or 4 (RGBA) channels")


def _rgb_to_yuv444_planar(frame: torch.Tensor) -> torch.Tensor:
    """Convert [H,W,3] RGB tensor (0..1 float or 0..255) to planar YUV444 Byte layout."""
    if not isinstance(frame, torch.Tensor):
        frame = torch.as_tensor(frame)
    if frame.dim() != 3 or frame.shape[2] != 3:
        raise ValueError("Expected RGB tensor with shape [H,W,3]")
    if frame.dtype != torch.float32:
        work = frame.to(dtype=torch.float32)
    else:
        work = frame
    if torch.max(work) <= 1.0:
        work = work * 255.0
    r = work[..., 0]
    g = work[..., 1]
    b = work[..., 2]

    # BT.709 conversion matrix for full-range YUV.
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = -0.1146 * r - 0.3854 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.4542 * g - 0.0458 * b + 128.0

    stacked = torch.stack([y, u, v], dim=0)
    stacked = torch.clamp(stacked, 0.0, 255.0).round().to(dtype=torch.uint8)
    planar = stacked.reshape(3 * frame.shape[0], frame.shape[1]).contiguous()
    return planar


def _rgb_to_nv12_planar(frame: torch.Tensor) -> torch.Tensor:
    if not isinstance(frame, torch.Tensor):
        frame = torch.as_tensor(frame)
    if frame.dim() != 3 or frame.shape[2] != 3:
        raise ValueError("Expected RGB tensor with shape [H,W,3]")
    device = frame.device
    work = frame.to(dtype=torch.float32)
    if torch.max(work) <= 1.0:
        work = work * 255.0
    r = work[..., 0]
    g = work[..., 1]
    b = work[..., 2]

    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = -0.1146 * r - 0.3854 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.4542 * g - 0.0458 * b + 128.0

    y_plane = torch.clamp(y, 0.0, 255.0).round().to(dtype=torch.uint8)
    h, w = y_plane.shape
    if h % 2:
        y_plane = torch.cat([y_plane, y_plane[-1:, :]], dim=0)
        u = torch.cat([u, u[-1:, :]], dim=0)
        v = torch.cat([v, v[-1:, :]], dim=0)
        h += 1
    if w % 2:
        y_plane = torch.cat([y_plane, y_plane[:, -1:].clone()], dim=1)
        u = torch.cat([u, u[:, -1:].clone()], dim=1)
        v = torch.cat([v, v[:, -1:].clone()], dim=1)
        w += 1

    u_down = (
        u[0::2, 0::2]
        + u[0::2, 1::2]
        + u[1::2, 0::2]
        + u[1::2, 1::2]
    ) * 0.25
    v_down = (
        v[0::2, 0::2]
        + v[0::2, 1::2]
        + v[1::2, 0::2]
        + v[1::2, 1::2]
    ) * 0.25

    u_down = torch.clamp(u_down, 0.0, 255.0).round().to(dtype=torch.uint8)
    v_down = torch.clamp(v_down, 0.0, 255.0).round().to(dtype=torch.uint8)

    uv_plane = torch.empty((h // 2, w), dtype=torch.uint8, device=device)
    uv_plane[:, 0::2] = u_down
    uv_plane[:, 1::2] = v_down

    planar = torch.cat((y_plane, uv_plane), dim=0).contiguous()
    return planar


class LosslessEncoder:
    def __init__(self, width: int, height: int, codec: str, fps: float) -> None:
        self.width = width
        self.height = height
        self.codec = codec
        self.fps = fps
        self._ticks = _ticks_per_frame(fps)
        self._sequence = 0
        self.hardware = False
        self.detail = "software"
        self.profile_level_id = "42e01f"

    def encode(
        self, frame: torch.Tensor, viewport: ViewportState
    ) -> Union[av.Packet, VideoFrame]:
        raise NotImplementedError

    def next_pts(self) -> int:
        value = self._sequence * self._ticks if self._ticks else self._sequence
        self._sequence += 1
        return value


class PyNvVideoCodecEncoder(LosslessEncoder):
    def __init__(
        self,
        width: int,
        height: int,
        codec: str,
        fps: float,
        pixel_format: str = "YUV444",
    ) -> None:
        super().__init__(width, height, codec, fps)
        if not torch.cuda.is_available():
            raise EncoderUnavailable("CUDA device required for NVENC encoding")
        nvc = _load_pynvcodec()
        codec_name = codec.lower()
        if codec_name not in {"h264", "av1", "hevc"}:
            raise ValueError(f"Unsupported codec {codec}")

        self._pixel_format = pixel_format.upper()
        frame_rate = Fraction(max(fps, 1.0)).limit_denominator(1000)
        gop = max(1, int(round(fps)) if fps > 0 else 1)
        gpu_id = torch.cuda.current_device()
        profile = None
        if codec_name == "h264":
            if self._pixel_format == "YUV444":
                profile = "high444p"
                self.profile_level_id = "f4001f"
            else:
                profile = "high"
                self.profile_level_id = "42e01f"
        elif codec_name == "hevc":
            profile = "main444-10" if self._pixel_format == "YUV444" else "main"
        elif codec_name == "av1":
            profile = None

        encoder_kwargs = dict(
            codec=codec_name,
            constqp=0,
            enableLossless=1,
            frameRateNum=frame_rate.numerator,
            frameRateDen=frame_rate.denominator,
            gopLength=gop,
            profile=profile,
            rateControlMode="constqp",
            gpuid=gpu_id,
        )
        encoder_kwargs = {k: v for k, v in encoder_kwargs.items() if v is not None}

        create_sig = inspect.signature(nvc.CreateEncoder)
        try:
            if "fmt" in create_sig.parameters:
                self._encoder = nvc.CreateEncoder(
                    width,
                    height,
                    self._pixel_format,
                    False,
                    **encoder_kwargs,
                )
            else:
                self._encoder = nvc.CreateEncoder(
                    width=width,
                    height=height,
                    format=self._pixel_format,
                    usecpuinputbuffer=False,
                    **encoder_kwargs,
                )
        except Exception as exc:
            raise EncoderUnavailable(f"NVENC initialization failed: {exc}") from exc
        self._nvc = nvc
        self._codec_name = codec_name
        self._force_idr = True
        self._pic_flags = getattr(nvc, "NV_ENC_PIC_FLAGS", None)
        self._device = torch.device("cuda", gpu_id)
        self.hardware = True
        self.detail = f"nvenc-{codec_name}-{self._pixel_format.lower()}"
        self._sprop_parameter_sets: Optional[str] = None
        self._prime_sprop()

    def encode(
        self, frame: torch.Tensor, viewport: ViewportState
    ) -> av.Packet:
        rgb = _ensure_rgb_tensor(frame)
        if self._pixel_format == "YUV444":
            planar = _rgb_to_yuv444_planar(rgb)
        else:
            planar = _rgb_to_nv12_planar(rgb)
        if planar.device != self._device:
            planar = planar.to(device=self._device, non_blocking=True)
        if not planar.is_contiguous():
            planar = planar.contiguous()

        pic_flags = 0
        if self._force_idr and self._pic_flags is not None:
            pic_flags |= int(self._pic_flags.FORCEIDR)
            pic_flags |= int(self._pic_flags.OUTPUT_SPSPPS)

        bitstream = self._encoder.Encode(planar, pic_flags) if pic_flags else self._encoder.Encode(planar)
        tail = self._encoder.EndEncode()
        if tail:
            bitstream = (bitstream or b"") + tail
        self._force_idr = False

        if not bitstream:
            raise RuntimeError("NVENC encoder produced empty bitstream")

        self._ensure_sprop(bitstream)

        packet = av.Packet(bitstream)
        if pic_flags:
            try:
                packet.is_keyframe = True
            except AttributeError:
                pass
        packet.pts = self.next_pts()
        packet.time_base = VIDEO_TIME_BASE
        packet.dts = packet.pts
        return packet

    def finish(self) -> None:
        try:
            tail = self._encoder.EndEncode()
            if tail:
                logger.debug("NVENC flushed %d trailing bytes", len(tail))
        except Exception:  # pragma: no cover - best effort
            logger.exception("Failed to flush NVENC encoder")

    @property
    def sprop_parameter_sets(self) -> Optional[str]:
        return self._sprop_parameter_sets

    def force_idr(self) -> None:
        self._force_idr = True

    def _ensure_sprop(self, bitstream: bytes) -> None:
        if self._sprop_parameter_sets is not None:
            return
        sps = None
        pps = None
        for nalu in _split_annexb_nalus(bitstream):
            if not nalu:
                continue
            nal_type = nalu[0] & 0x1F
            if nal_type == 7 and sps is None:
                sps = nalu
            elif nal_type == 8 and pps is None:
                pps = nalu
            if sps is not None and pps is not None:
                break
        if sps and pps:
            sps_b64 = base64.b64encode(sps).decode("ascii")
            pps_b64 = base64.b64encode(pps).decode("ascii")
            self._sprop_parameter_sets = f"{sps_b64},{pps_b64}"

    def _prime_sprop(self) -> None:
        try:
            frame = torch.zeros(
                (self.height, self.width, 3),
                dtype=torch.float32,
                device=self._device,
            )
            if self._pixel_format == "YUV444":
                planar = _rgb_to_yuv444_planar(frame)
            else:
                planar = _rgb_to_nv12_planar(frame)
            flags = 0
            if self._pic_flags is not None:
                flags |= int(self._pic_flags.FORCEIDR)
                flags |= int(self._pic_flags.OUTPUT_SPSPPS)
            bitstream = self._encoder.Encode(planar, flags)
            tail = self._encoder.EndEncode()
            if tail:
                bitstream = (bitstream or b"") + tail
            if bitstream:
                self._ensure_sprop(bitstream)
        except Exception:
            logger.debug("Failed to prime NVENC SPS/PPS", exc_info=True)
        finally:
            self._force_idr = True


def _split_annexb_nalus(data: bytes) -> list[bytes]:
    nalus: list[bytes] = []
    i = 0
    start = None
    length = len(data)
    while i < length - 3:
        if data[i : i + 4] == b"\x00\x00\x00\x01":
            if start is not None and start < i:
                nalus.append(data[start:i])
            start = i + 4
            i += 4
        elif data[i : i + 3] == b"\x00\x00\x01":
            if start is not None and start < i:
                nalus.append(data[start:i])
            start = i + 3
            i += 3
        else:
            i += 1
    if start is not None and start < length:
        nalus.append(data[start:])
    return nalus


def _annexb_to_length_prefixed(data: bytes) -> bytes:
    nalus = _split_annexb_nalus(data)
    output = bytearray()
    for nalu in nalus:
        output += len(nalu).to_bytes(4, byteorder="big")
        output += nalu
    return bytes(output)


    def _prime_sprop(self) -> None:
        try:
            frame = torch.zeros(
                (self.height, self.width, 3),
                dtype=torch.float32,
                device=self._device,
            )
            if self._pixel_format == "YUV444":
                planar = _rgb_to_yuv444_planar(frame)
            else:
                planar = _rgb_to_nv12_planar(frame)
            flags = 0
            if self._pic_flags is not None:
                flags |= int(self._pic_flags.FORCEIDR)
                if hasattr(self._pic_flags, "OUTPUT_SPSPPS"):
                    flags |= int(self._pic_flags.OUTPUT_SPSPPS)
            bitstream = self._encoder.Encode(planar, flags)
            tail = self._encoder.EndEncode()
            if tail:
                bitstream = (bitstream or b"") + tail
            if bitstream:
                self._ensure_sprop(bitstream)
        except Exception:  # pragma: no cover - best effort
            logger.debug("Failed to prime NVENC SPS/PPS", exc_info=True)
        finally:
            self._force_idr = True


class SoftwareLosslessEncoder(LosslessEncoder):
    """CPU fallback, primarily for testing."""

    def __init__(self, width: int, height: int, codec: str, fps: float) -> None:
        super().__init__(width, height, codec, fps)
        self.detail = "software-pyav"
        self.profile_level_id = "42e01f"

    def encode(
        self, frame: torch.Tensor, viewport: ViewportState
    ) -> VideoFrame:
        rgb = _ensure_rgb_tensor(frame)
        rgb_cpu = rgb.detach().cpu()
        if rgb_cpu.dtype != torch.uint8:
            work = rgb_cpu.to(dtype=torch.float32)
            if torch.max(work) <= 1.0:
                work = work * 255.0
            work = torch.clamp(work, 0.0, 255.0).to(dtype=torch.uint8)
            rgb_cpu = work
        rgb_array = rgb_cpu.numpy()
        video_frame = VideoFrame.from_ndarray(rgb_array, format="rgb24")
        pts = self.next_pts()
        if self.fps > 0:
            video_frame.pts = pts
            video_frame.time_base = Fraction(1, max(1, int(round(self.fps))))
        return video_frame


def create_encoder(
    width: int,
    height: int,
    codec: str,
    fps: float,
    prefer_hardware: bool,
    pixel_format: str = "NV12",
) -> LosslessEncoder:
    codec = codec.lower()
    if codec in {"av1", "hevc"}:
        # aiortc lacks built-in packetizers; instruct user for now.
        raise EncoderUnavailable(
            f"Codec '{codec}' is not yet supported by the TorchWindow SFU pipeline."
            " Please choose codec='h264' until AV1/HEVC packetizers are available."
        )
    if prefer_hardware:
        try:
            return PyNvVideoCodecEncoder(
                width, height, codec, fps, pixel_format=pixel_format
            )
        except EncoderUnavailable as exc:
            logger.warning(
                "NVENC unavailable, falling back to software encoder: %s", exc
            )
            _log_diagnostics_warning()
    return SoftwareLosslessEncoder(width, height, codec, fps)


def _log_diagnostics_warning() -> None:
    try:
        from .diagnostics import run_diagnostics

        report = run_diagnostics()
        for check in report.get("checks", []):
            name = check.get("name", "unknown")
            status = check.get("status", "unknown").upper()
            detail = check.get("detail", "")
            logger.warning("Diagnostic %s [%s]: %s", name, status, detail)
        for advice in report.get("advice", []):
            logger.warning("Diagnostic advice: %s", advice)
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.debug("Failed to run diagnostics: %s", exc, exc_info=True)
