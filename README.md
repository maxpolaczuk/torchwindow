# TorchWindow

TorchWindow is a Python library that enables viewing of PyTorch Cuda Tensors on screen directly from GPU memory (No copying back and forth between GPU and CPU) via OpenGL-Cuda interop.

## Install

```
pip install torchwindow
```

## Use
To create a window
```
from torchwindow import Window
window = Window(640, 480, name="Torch Window")
```
To display an image tensor in the window
```
window.draw(image)
```
`image` must be a tensor with the following properties:
- 3 dimensions, specifically `(rows, columns, channels)` in that order.
- `channels` dimension must be of size 4 (r, g, b, a)

## Streaming Diagnostics

TorchWindow 2.0 includes a diagnostic command to check for CUDA, NVENC, and driver readiness.

```
torchwindow diagnose
```

Use the output to confirm that `PyNvVideoCodec` is installed and that your environment exposes NVIDIA NVENC hardware.
