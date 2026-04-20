"""SDL2 + Metal presentation path for Apple Silicon / PyTorch MPS.

Why Metal and not GL on macOS: Apple deprecated OpenGL (capped at 4.1,
no interop with anything modern). PyTorch-MPS tensors live in Metal
heaps; keeping the presentation in Metal lets us wrap their backing
memory directly into an MTLTexture and avoid every CPU round-trip.

This module uses three Python ↔ Objective-C bridges:
- ``pysdl2`` for window/event-loop (same as the GL path).
- ``Metal``, ``QuartzCore``, ``Cocoa`` via ``pyobjc`` for Metal APIs.
- ``ctypes`` for the two SDL Metal entry points (``SDL_Metal_CreateView``
  and ``SDL_Metal_GetLayer``) that pysdl2 0.9 doesn't expose directly.
"""

from __future__ import annotations

import ctypes
import logging
import warnings
from typing import Any, Optional

with warnings.catch_warnings():
    warnings.filterwarnings(action="ignore", category=UserWarning)
    import sdl2

from .._normalize import to_canonical
from .._validate import validate_tensor
from ..backends.metal import MetalBackend
from ..exceptions import SDLException
from ..metal_shaders import MSL_SOURCE

logger = logging.getLogger(__name__)


# --- SDL Metal C entry points via ctypes ---------------------------------
# pysdl2 wraps most of SDL but not the Metal helpers on the version we pin.
# We resolve the symbols from the already-loaded libSDL2 and call them
# ourselves. ``_sdl_dll()`` finds the library by asking ctypes for the
# handle underneath ``sdl2.dll`` (pysdl2's internal reference).

def _sdl_dll() -> ctypes.CDLL:
    # pysdl2 stores the loaded CDLL at sdl2.dll.dll._dll (module -> DLL
    # wrapper -> ctypes handle). Falling back to find_library would
    # resolve a *different* SDL2 on the system and segfault.
    dll_module = getattr(sdl2, "dll", None)
    dll_wrapper = getattr(dll_module, "dll", None)
    cdll = getattr(dll_wrapper, "_dll", None)
    if cdll is not None:
        return cdll  # type: ignore[no-any-return]
    import ctypes.util

    return ctypes.CDLL(ctypes.util.find_library("SDL2"))


def _bind_metal_funcs() -> None:
    lib = _sdl_dll()
    lib.SDL_Metal_CreateView.restype = ctypes.c_void_p
    lib.SDL_Metal_CreateView.argtypes = [ctypes.c_void_p]
    lib.SDL_Metal_GetLayer.restype = ctypes.c_void_p
    lib.SDL_Metal_GetLayer.argtypes = [ctypes.c_void_p]
    lib.SDL_Metal_DestroyView.restype = None
    lib.SDL_Metal_DestroyView.argtypes = [ctypes.c_void_p]


# SDL_WINDOW_METAL = 0x20000000 (SDL 2.0.14+); hard-code in case pysdl2's
# constant set is older than the runtime libSDL2.
SDL_WINDOW_METAL = 0x20000000


class MetalWindow:
    """SDL2 + Metal presentation path. Paired with ``MetalBackend``."""

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        name: str = "torchwindow",
        backend: Optional[str] = None,  # noqa: ARG002 — kept for API parity
    ) -> None:
        import Metal  # type: ignore[import-not-found]
        import objc  # type: ignore[import-not-found]

        self._Metal = Metal
        self._objc = objc
        self.name = name
        self.width = width
        self.height = height
        self.running = True
        self._closed = False

        self.sdl_window: Any = None
        self._metal_view: Optional[int] = None
        self._layer: Any = None
        self._device: Any = None
        self._queue: Any = None
        self._pipeline: Any = None
        self._sampler: Any = None
        self.backend: MetalBackend = MetalBackend()

        self.setup()

    # -- init helpers -----------------------------------------------------
    def setup(self) -> None:
        self._setup_sdl()
        self._setup_metal()
        self.backend.register(self._device, self.width, self.height)

    def _setup_sdl(self) -> None:
        _bind_metal_funcs()
        if sdl2.SDL_Init(sdl2.SDL_INIT_VIDEO):
            raise SDLException(sdl2.SDL_GetError())

        self.sdl_window = sdl2.SDL_CreateWindow(
            self.name.encode(),
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            sdl2.SDL_WINDOWPOS_UNDEFINED,
            self.width,
            self.height,
            SDL_WINDOW_METAL | sdl2.SDL_WINDOW_RESIZABLE,
        )
        if not self.sdl_window:
            raise SDLException(sdl2.SDL_GetError())

        lib = _sdl_dll()
        win_ptr = ctypes.cast(self.sdl_window, ctypes.c_void_p)
        self._metal_view = lib.SDL_Metal_CreateView(win_ptr)
        if not self._metal_view:
            raise SDLException(sdl2.SDL_GetError())
        layer_ptr = lib.SDL_Metal_GetLayer(ctypes.c_void_p(self._metal_view))
        if not layer_ptr:
            raise SDLException(sdl2.SDL_GetError())
        self._layer = self._objc.objc_object(c_void_p=layer_ptr)

    def _setup_metal(self) -> None:
        Metal = self._Metal

        self._device = Metal.MTLCreateSystemDefaultDevice()
        if self._device is None:
            raise RuntimeError(
                "MTLCreateSystemDefaultDevice returned nil — no Metal GPU?"
            )

        self._layer.setDevice_(self._device)
        self._layer.setPixelFormat_(Metal.MTLPixelFormatBGRA8Unorm)
        self._layer.setFramebufferOnly_(True)

        self._queue = self._device.newCommandQueue()

        options = Metal.MTLCompileOptions.alloc().init()
        library, err = self._device.newLibraryWithSource_options_error_(
            MSL_SOURCE, options, None
        )
        if library is None:
            raise RuntimeError(f"Metal shader compilation failed: {err}")
        vs = library.newFunctionWithName_("vs_main")
        fs = library.newFunctionWithName_("fs_main")

        pdesc = Metal.MTLRenderPipelineDescriptor.alloc().init()
        pdesc.setVertexFunction_(vs)
        pdesc.setFragmentFunction_(fs)
        pdesc.colorAttachments().objectAtIndexedSubscript_(0).setPixelFormat_(
            Metal.MTLPixelFormatBGRA8Unorm
        )
        self._pipeline, err = self._device.newRenderPipelineStateWithDescriptor_error_(
            pdesc, None
        )
        if self._pipeline is None:
            raise RuntimeError(f"Render pipeline creation failed: {err}")

        sdesc = Metal.MTLSamplerDescriptor.alloc().init()
        sdesc.setMinFilter_(Metal.MTLSamplerMinMagFilterLinear)
        sdesc.setMagFilter_(Metal.MTLSamplerMinMagFilterLinear)
        sdesc.setSAddressMode_(Metal.MTLSamplerAddressModeRepeat)
        sdesc.setTAddressMode_(Metal.MTLSamplerAddressModeRepeat)
        self._sampler = self._device.newSamplerStateWithDescriptor_(sdesc)

    # -- per-frame --------------------------------------------------------
    def draw(self, tensor: Any, stream: Any = None) -> None:
        if not self.running:
            return
        tensor = to_canonical(tensor)
        validate_tensor(tensor, self.width, self.height)
        self.backend.upload(tensor, self.width, self.height, stream=stream)
        self._step_events()
        if self.running:
            self._render()

    def _render(self) -> None:
        Metal = self._Metal
        drawable = self._layer.nextDrawable()
        if drawable is None:
            return  # surface not ready; skip frame

        texture = self.backend.current_texture()
        if texture is None:
            return

        rpd = Metal.MTLRenderPassDescriptor.renderPassDescriptor()
        color = rpd.colorAttachments().objectAtIndexedSubscript_(0)
        color.setTexture_(drawable.texture())
        color.setLoadAction_(Metal.MTLLoadActionClear)
        color.setStoreAction_(Metal.MTLStoreActionStore)
        color.setClearColor_(Metal.MTLClearColorMake(0.0, 0.0, 0.0, 1.0))

        cmd = self._queue.commandBuffer()
        enc = cmd.renderCommandEncoderWithDescriptor_(rpd)
        enc.setRenderPipelineState_(self._pipeline)
        enc.setFragmentTexture_atIndex_(texture, 0)
        enc.setFragmentSamplerState_atIndex_(self._sampler, 0)
        enc.drawPrimitives_vertexStart_vertexCount_(
            Metal.MTLPrimitiveTypeTriangle, 0, 3
        )
        enc.endEncoding()
        cmd.presentDrawable_(drawable)
        cmd.commit()

    def _step_events(self) -> None:
        event = sdl2.SDL_Event()
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

    def _resize(self, width: int, height: int) -> None:
        if width == self.width and height == self.height:
            return
        self.width = width
        self.height = height
        self.backend.unregister()
        self.backend.register(self._device, width, height)
        # CAMetalLayer auto-resizes to the view's bounds; no further action.

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.running = False
        try:
            self.backend.unregister()
        finally:
            if self._metal_view:
                _sdl_dll().SDL_Metal_DestroyView(
                    ctypes.c_void_p(self._metal_view)
                )
                self._metal_view = None
            if self.sdl_window is not None:
                sdl2.SDL_DestroyWindow(self.sdl_window)
                self.sdl_window = None
            sdl2.SDL_Quit()

    def __enter__(self) -> "MetalWindow":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
