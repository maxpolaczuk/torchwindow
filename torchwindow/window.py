import ctypes
import warnings
from pathlib import Path
from typing import Optional, Tuple

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=UserWarning)
    import sdl2

from sdl2 import video
from OpenGL import GL as gl
from cuda import cudart as cu

import imageio.v2 as imageio
import numpy as np

from .shaders import create_shader_program
from .exceptions import SDLException, CudaException, OpenGLException
try:
    from .streaming import StreamConfig, WebRTCStreamer
except ImportError:  # pragma: no cover - optional dependency path
    StreamConfig = None
    WebRTCStreamer = None

import logging

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_FPS = 30


class Window:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        name: str = "torchwindow",
        save_output_dir: Optional[Path] = None,
        lossyness: float = 0.0,
        stream_enabled: bool = False,
        stream_only: bool = False,
        stream_host: str = "0.0.0.0",
        stream_port: int = 8765,
        stream_queue_size: int = 3,
        stream_max_bitrate: Optional[int] = None,
        stream_min_bitrate: Optional[int] = None,
        stream_max_framerate: Optional[float] = None,
        stream_codec: Optional[str] = None,
    ):
        self.name = name
        self.width = width
        self.height = height

        self.cuda_is_setup = False
        self.running = True

        self.stream_only = stream_only
        self.stream_enabled = stream_enabled or stream_only
        self.stream_host = stream_host
        self.stream_port = stream_port
        self.stream_queue_size = stream_queue_size
        self.stream_max_bitrate = stream_max_bitrate
        self.stream_min_bitrate = stream_min_bitrate
        self.stream_max_framerate = (
            stream_max_framerate if stream_max_framerate is not None else DEFAULT_FALLBACK_FPS
        )
        self.stream_codec = stream_codec
        self.streamer = None

        self.output_mode = "stream" if self.stream_only else "window"
        self.save_output_dir = Path(save_output_dir).expanduser() if save_output_dir else None
        self.lossyness = float(lossyness)
        if self.lossyness < 0:
            raise ValueError("lossyness must be non-negative")
        self.lossyness = min(self.lossyness, 1.0)

        self._video_writer = None
        self._output_path = None
        self._frame_count = 0
        self._sdl_initialized = False
        self._fallback_cause = None

        if self.output_mode == "stream":
            self._ensure_streamer()
        else:
            self.setup()

    def setup_sdl(self):
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO):
            raise SDLException(sdl2.SDL_GetError())

        self._sdl_initialized = True

        self.sdl_window = sdl2.SDL_CreateWindow(
            self.name.encode(),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            self.width,
            self.height,
            sdl2.SDL_WINDOW_OPENGL,
        )
        if not self.sdl_window:
            raise SDLException(sdl2.SDL_GetError())

        # Force OpenGL 3.3 'core' context.
        # Must set *before* creating GL context!
        video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MINOR_VERSION, 3)
        video.SDL_GL_SetAttribute(
            video.SDL_GL_CONTEXT_PROFILE_MASK, video.SDL_GL_CONTEXT_PROFILE_CORE
        )
        self.gl_context = sdl2.SDL_GL_CreateContext(self.sdl_window)

    def setup_opengl(self):
        self.shader_program = create_shader_program()
        self.vao = gl.glGenVertexArrays(1)

        self.tex = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            self.width,
            self.height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def setup_cuda(self):
        if self.cuda_is_setup:
            return

        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO) != 0:
            raise SDLException(sdl2.SDL_GetError())

        err, *_ = cu.cudaGLGetDevices(1, cu.cudaGLDeviceList.cudaGLDeviceListAll)
        if err == cu.cudaError_t.cudaErrorUnknown:
            raise OpenGLException(
                "OpenGL context may be running on integrated graphics"
            )

        err, self.cuda_image = cu.cudaGraphicsGLRegisterImage(
            self.tex,
            gl.GL_TEXTURE_2D,
            cu.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to register opengl texture")

        self.cuda_is_setup = True

    def render(self):
        gl.glUseProgram(self.shader_program)
        try:
            gl.glClearColor(0, 0, 0, 1)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        finally:
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
        sdl2.SDL_GL_SwapWindow(self.sdl_window)

    def step(self):
        event = sdl2.SDL_Event()
        if self.running:
            while sdl2.SDL_PollEvent(ctypes.byref(event)):
                if (
                    event.type == sdl2.SDL_WINDOWEVENT
                    and event.window.event == sdl2.SDL_WINDOWEVENT_CLOSE
                ):
                    self.running = False
            self.render()

    def setup(self):
        try:
            self.setup_sdl()
            self.setup_opengl()
            self.setup_cuda()
        except (SDLException, OpenGLException, CudaException) as exc:
            self._activate_fallback(exc)

    def draw(self, tensor):
        if self.output_mode == "stream":
            self._write_stream_frame(tensor)
            return
        if self.output_mode == "file":
            self._write_frame(tensor)
            return

        if not self.running:
            return

        try:
            self._draw_gpu(tensor)
        except (CudaException, OpenGLException) as exc:
            mode = self._activate_fallback(exc)
            if mode == "file":
                self._write_frame(tensor)
            elif mode == "stream":
                self._write_stream_frame(tensor)

    def _draw_gpu(self, tensor):
        if not self.cuda_is_setup:
            self.setup_cuda()
        (err,) = cu.cudaGraphicsMapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to map graphics resource")
        err, array = cu.cudaGraphicsSubResourceGetMappedArray(self.cuda_image, 0, 0)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to get mapped array")
        (err,) = cu.cudaMemcpy2DToArrayAsync(
            array,
            0,
            0,
            tensor.data_ptr(),
            4 * 4 * self.width,
            4 * 4 * self.width,
            self.height,
            cu.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
            cu.cudaStreamLegacy,
        )
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to copy from tensor to texture")

        (err,) = cu.cudaGraphicsUnmapResources(1, self.cuda_image, cu.cudaStreamLegacy)
        if err != cu.cudaError_t.cudaSuccess:
            raise CudaException("Unable to unmap graphics resource")
        self.step()

    def _fallback_to_file_output(self, cause: Exception):
        if self.save_output_dir is None:
            raise RuntimeError(
                "CUDA/OpenGL interop is unavailable on this platform. "
                "Provide save_output_dir when creating Window to enable file output fallback."
            ) from cause

        if self.output_mode == "file":
            return

        logger.warning("Falling back to video output because window rendering failed: %s", cause)
        self._cleanup_graphics()

        self.output_mode = "file"
        self.running = True
        self.cuda_is_setup = False
        self.save_output_dir.mkdir(parents=True, exist_ok=True)
        self._video_writer = None
        self._output_path = None
        self._frame_count = 0
        self._fallback_cause = cause

    def _fallback_to_stream(self, cause: Exception):
        if not self.stream_enabled:
            raise RuntimeError(
                "Streaming fallback requested but stream_enabled=False."
            ) from cause

        if self.output_mode == "stream":
            return

        logger.warning(
            "Falling back to WebRTC streaming because window rendering failed: %s",
            cause,
        )
        self._cleanup_graphics()
        self._ensure_streamer()
        self.output_mode = "stream"
        self.running = True
        self.cuda_is_setup = False
        self._fallback_cause = cause

    def _write_frame(self, tensor):
        frame = self._tensor_to_frame(tensor)
        self._ensure_video_writer(frame.shape)
        self._video_writer.append_data(frame)
        self._frame_count += 1

    def _write_stream_frame(self, tensor):
        frame = self._tensor_to_frame(tensor)
        self._ensure_streamer()
        self.streamer.send(frame)

    def _ensure_video_writer(self, frame_shape: Tuple[int, int, int]):
        if self._video_writer is not None:
            return

        extension, writer_args = self._encoding_profile()
        output_path = self.save_output_dir / f"{self.name}{extension}"

        writer_options = dict(writer_args)
        writer_options.setdefault("format", "FFMPEG")
        writer_options.setdefault("fps", DEFAULT_FALLBACK_FPS)
        writer_options.setdefault("macro_block_size", None)

        self._video_writer = imageio.get_writer(output_path, **writer_options)
        self._output_path = output_path
        logger.info(
            "Writing %dx%d frames to %s",
            frame_shape[1],
            frame_shape[0],
            output_path,
        )

    def _encoding_profile(self) -> Tuple[str, dict]:
        level = max(0.0, min(1.0, self.lossyness))
        crf = int(round(level * 35))
        ffmpeg_params = ["-crf", str(crf), "-preset", "medium", "-pix_fmt", "rgb24"]
        codec = "libx264rgb"
        extension = ".mp4"
        return extension, {"codec": codec, "ffmpeg_params": ffmpeg_params}

    def _tensor_to_frame(self, tensor) -> np.ndarray:
        cpu_tensor = tensor.detach().to("cpu")
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()
        array = cpu_tensor.numpy()
        if array.ndim != 3:
            raise ValueError("Expected tensor with shape (H, W, C)")

        # Promote values to float32 for consistent scaling.
        if array.dtype != np.float32:
            array = array.astype(np.float32)

        channels = array.shape[-1]
        if channels == 1:
            array = np.repeat(array, 3, axis=-1)
        elif channels == 4:
            array = array[..., :3]
        elif channels != 3:
            raise ValueError("Expected tensor with 1, 3, or 4 channels")

        array = np.clip(array, 0.0, 1.0)
        frame = np.rint(array * 255.0).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def _ensure_streamer(self):
        if not self.stream_enabled:
            raise RuntimeError("Streaming not enabled for this window")
        if self.streamer is None:
            if WebRTCStreamer is None or StreamConfig is None:
                raise RuntimeError(
                    "WebRTC dependencies are not installed. Install torchwindow with streaming extras."
                )
            max_br, min_br = self._stream_bitrate_params()
            config = StreamConfig(
                host=self.stream_host,
                port=self.stream_port,
                queue_size=self.stream_queue_size,
                max_bitrate=max_br,
                min_bitrate=min_br,
                max_framerate=self.stream_max_framerate,
                codec=self.stream_codec,
            )
            self.streamer = WebRTCStreamer(config, self.name)
            self.streamer.start()

    def _activate_fallback(self, cause: Exception) -> str:
        if self.stream_enabled:
            self._fallback_to_stream(cause)
            return "stream"
        if self.save_output_dir is not None:
            self._fallback_to_file_output(cause)
            return "file"
        self._fallback_to_file_output(cause)
        return "file"

    def _stream_bitrate_params(self) -> Tuple[int, int]:
        if self.stream_max_bitrate is not None:
            max_br = int(self.stream_max_bitrate)
        else:
            quality = 1.0 - self.lossyness
            max_br = int(1_500_000 + quality * 8_500_000)
        if self.stream_min_bitrate is not None:
            min_br = int(self.stream_min_bitrate)
        else:
            min_br = max(100_000, int(max_br * 0.5))
        return max_br, min_br

    def _cleanup_graphics(self):
        if getattr(self, "cuda_image", None) is not None and self.cuda_is_setup:
            (err,) = cu.cudaGraphicsUnregisterResource(self.cuda_image)
            if err != cu.cudaError_t.cudaSuccess:
                logger.warning("Unable to unregister CUDA graphics resource: %s", err)
            self.cuda_image = None
            self.cuda_is_setup = False

        if getattr(self, "gl_context", None):
            sdl2.SDL_GL_DeleteContext(self.gl_context)
            self.gl_context = None
        if getattr(self, "sdl_window", None):
            sdl2.SDL_DestroyWindow(self.sdl_window)
            self.sdl_window = None
        if self._sdl_initialized:
            sdl2.SDL_Quit()
            self._sdl_initialized = False

    def _finalize_file_output(self):
        if self._video_writer is None:
            return
        self._video_writer.close()
        logger.info("Saved %d frame(s) to %s", self._frame_count, self._output_path)
        self._video_writer = None

    def close(self):
        self.running = False
        self._finalize_file_output()
        self._cleanup_graphics()
        if self.streamer is not None:
            self.streamer.close()
