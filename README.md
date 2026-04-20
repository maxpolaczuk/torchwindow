# TorchWindow

Display PyTorch tensors directly from GPU memory — in a desktop window, or
inside a Jupyter notebook. Minimizes host↔device copies so the visualizer
keeps up with your training loop instead of stalling it.

Backends:

| Platform | Backend | Transfer path |
| --- | --- | --- |
| Linux / Windows (NVIDIA) | **OpenGL + CUDA** | Device-to-device via `cudaGraphicsGLRegisterImage` — zero CPU involvement. |
| macOS (Apple Silicon) | **Metal + PyTorch MPS** | Host roundtrip today; true zero-copy is a tracked follow-up (see below). |
| Jupyter (any platform) | **anywidget `NotebookWindow`** | One host copy per frame, then uint8 RGBA over the Jupyter comm channel. |

## Install

```bash
pip install torchwindow                 # CUDA desktop window
pip install 'torchwindow[mps]'          # + macOS Metal path
pip install 'torchwindow[notebook]'     # + Jupyter NotebookWindow
```

## Desktop

```python
import torch
from torchwindow import Window

window = Window(640, 480, name="Torch Window")
image = torch.rand(480, 640, 4, device="cuda")    # or device="mps" on macOS
window.draw(image)
```

`Window(...)` picks the Metal path on macOS and the CUDA/GL path elsewhere.
Override with `backend="cuda"` or `backend="metal"`.

## Jupyter

```python
import torch
from torchwindow import NotebookWindow

w = NotebookWindow(512, 512)
w                                       # renders the widget inline

for step in range(200):
    frame = torch.rand(512, 512, 4, device="mps")   # or "cuda"
    w.draw(frame)
```

## Accepted tensor shapes and dtypes

Everything passes through a normalizer that converts to the canonical
`(H, W, 4) float32` layout *on the tensor's device* before upload. Accepted
inputs:

- Shapes: `(H, W, 4)`, `(H, W, 3)` (alpha padded), `(H, W, 1)` (greyscale),
  `(C, H, W)` with `C ∈ {1, 3, 4}` (permuted).
- Dtypes: `float32` (pass-through), `float16` / `bfloat16` (upcast),
  `uint8` (scaled by 1/255).

Non-conforming shapes or dtypes raise a descriptive `ValueError`.

## Example

Runs a 4-quadrant color test image for a few seconds, using the best backend
available on the host:

```bash
python -m torchwindow.example                    # auto
python -m torchwindow.example --backend cuda     # force CUDA/GL
python -m torchwindow.example --backend metal    # force Metal/MPS
```

![Example](example.png)

## Roadmap

- **True zero-copy on MPS.** Requires access to the private `MTLBuffer`
  backing an MPS tensor, which PyTorch does not expose to Python today. Two
  viable paths: a tiny C++ extension calling
  `at::mps::getMTLBufferStorage`, or waiting for an official
  `torch.mps.buffer_of(tensor)` API.
- **Overlays.** `draw_boxes`, `draw_text`, `draw_mask` composited over the
  tensor texture in a second render pass.
- **Video recording.** `window.record("out.mp4")` — NVENC on CUDA,
  VideoToolbox on macOS, no CPU copy through the encoder.
- **Grid rendering.** `window.draw_grid(tensor_NHWC, ncols=4)` to tile a
  batch in one window.
