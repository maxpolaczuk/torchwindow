from __future__ import annotations

import ctypes
import logging
import warnings
from typing import Any, Optional

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=UserWarning)
    import sdl2

from OpenGL import GL as gl
from sdl2 import video

from . import backends as _backends
from ._validate import validate_tensor
from .backends.base import Backend
from .exceptions import SDLException
from .shaders import create_shader_program

logger = logging.getLogger(__name__)


class Window:
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        name: str = "torchwindow",
        backend: Optional[str] = None,
    ) -> None:
        self.name = name
        self.width = width
        self.height = height
        self._backend_name = backend

        self.running = True
        self._closed = False
        self.sdl_window = None
        self.gl_context = None
        self.tex: Optional[int] = None
        self.vao: Optional[int] = None
        self.shader_program: Optional[int] = None
        self.backend: Optional[Backend] = None

        self.setup()

    def setup_sdl(self) -> None:
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO):
            raise SDLException(sdl2.SDL_GetError())

        video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MAJOR_VERSION, 3)
        video.SDL_GL_SetAttribute(video.SDL_GL_CONTEXT_MINOR_VERSION, 3)
        video.SDL_GL_SetAttribute(
            video.SDL_GL_CONTEXT_PROFILE_MASK, video.SDL_GL_CONTEXT_PROFILE_CORE
        )

        self.sdl_window = sdl2.SDL_CreateWindow(
            self.name.encode(),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            self.width,
            self.height,
            sdl2.SDL_WINDOW_OPENGL | sdl2.SDL_WINDOW_RESIZABLE,
        )
        if not self.sdl_window:
            raise SDLException(sdl2.SDL_GetError())

        self.gl_context = sdl2.SDL_GL_CreateContext(self.sdl_window)
        if not self.gl_context:
            raise SDLException(sdl2.SDL_GetError())

    def setup_opengl(self) -> None:
        self.shader_program = create_shader_program()
        self.vao = gl.glGenVertexArrays(1)
        self.tex = gl.glGenTextures(1)
        self._allocate_texture_storage(self.width, self.height)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def _allocate_texture_storage(self, width: int, height: int) -> None:
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            None,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def setup(self) -> None:
        self.setup_sdl()
        self.setup_opengl()

    def _ensure_backend(self, tensor: Any) -> None:
        if self.backend is not None:
            return
        self.backend = _backends.resolve(self._backend_name, tensor)
        assert self.tex is not None
        self.backend.register(self.tex, self.width, self.height)

    def render(self) -> None:
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

    def _resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        if self.backend is not None:
            self.backend.unregister()
        self.width = width
        self.height = height
        self._allocate_texture_storage(width, height)
        gl.glViewport(0, 0, width, height)
        if self.backend is not None:
            assert self.tex is not None
            self.backend.register(self.tex, width, height)

    def step(self) -> None:
        event = sdl2.SDL_Event()
        if not self.running:
            return
        while sdl2.SDL_PollEvent(ctypes.byref(event)):
            if event.type == sdl2.SDL_WINDOWEVENT:
                if event.window.event == sdl2.SDL_WINDOWEVENT_CLOSE:
                    self.running = False
                elif event.window.event == sdl2.SDL_WINDOWEVENT_SIZE_CHANGED:
                    self._resize(event.window.data1, event.window.data2)
            elif event.type == sdl2.SDL_KEYDOWN:
                if event.key.keysym.sym == sdl2.SDLK_ESCAPE:
                    self.running = False
            elif event.type == sdl2.SDL_QUIT:
                self.running = False
        self.render()

    def draw(self, tensor: Any, stream: Any = None) -> None:
        if not self.running:
            return
        validate_tensor(tensor, self.width, self.height)
        self._ensure_backend(tensor)
        assert self.backend is not None
        self.backend.upload(tensor, self.width, self.height, stream=stream)
        self.step()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.running = False
        if self.backend is not None:
            self.backend.unregister()
            self.backend = None
        if self.gl_context is not None:
            sdl2.SDL_GL_DeleteContext(self.gl_context)
            self.gl_context = None
        if self.sdl_window is not None:
            sdl2.SDL_DestroyWindow(self.sdl_window)
            self.sdl_window = None
        sdl2.SDL_Quit()

    def __enter__(self) -> "Window":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
