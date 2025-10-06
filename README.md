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

### Headless or Virtualised Environments

On systems where CUDA/OpenGL interoperability cannot be used (for example headless servers or virtual machines), provide a directory for video capture when creating the window:

```
window = Window(
    640,
    480,
    name="Torch Window",
    save_output_dir="renders",
    lossyness=0,
)
```

If the graphics path fails, TorchWindow automatically switches to writing frames into `<save_output_dir>/<window name>.mp4`. The `lossyness` parameter controls the compression level of this video (0 is lossless, 1 is the highest compression). When no fallback directory is given and graphics interop fails, a `RuntimeError` is raised to prompt you to enable file output.

### Browser Streaming (WebRTC)

For truly headless usage, TorchWindow can expose the tensor as a live WebRTC stream with a bundled viewer:

```
window = Window(
    640,
    480,
    name="Torch Window",
    stream_enabled=True,
    stream_only=True,
    stream_host="0.0.0.0",
    stream_port=8765,
)
```

Navigate to `http://<stream_host>:<stream_port>/` and click **Connect** to watch the tensor in your browser. If `stream_only=False`, the window will prefer the native renderer and automatically fall back to WebRTC streaming when CUDA/OpenGL interop fails.

- **Quality control** – TorchWindow reuses the `lossyness` value for streaming. Set it to `0` (or supply `TORCHWINDOW_LOSSYNESS=0`) for the highest bitrate and minimal compression. You can override the encoder directly with `stream_max_bitrate`, `stream_min_bitrate`, or `stream_codec="H264"` when constructing the window (or via `TORCHWINDOW_STREAM_MAX_BITRATE`, `TORCHWINDOW_STREAM_MIN_BITRATE`, `TORCHWINDOW_STREAM_CODEC`).

## Example
To check if torchwindow is properly installed try running
```
python3 -m torchwindow.example
```
You should see this window appear for 5 seconds before closing
![Example](example.png)
