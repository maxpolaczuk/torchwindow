import math
import os
import time
from pathlib import Path
from typing import Optional

import torch

from torchwindow import Window


FPS = 30
DURATION_SECONDS = 30
WIDTH = 800
HEIGHT = 600


def build_coordinate_grid(device: torch.device):
    y = torch.linspace(0.0, 1.0, HEIGHT, device=device, dtype=torch.float32).unsqueeze(1)
    x = torch.linspace(0.0, 1.0, WIDTH, device=device, dtype=torch.float32).unsqueeze(0)
    x_grid = x.repeat(HEIGHT, 1)
    y_grid = y.repeat(1, WIDTH)
    return x_grid, y_grid


def render_frame(x_grid: torch.Tensor, y_grid: torch.Tensor, frame_idx: int) -> torch.Tensor:
    t = frame_idx / (FPS * DURATION_SECONDS)
    two_pi = 2.0 * math.pi

    top_mask = y_grid < 0.5

    phase = two_pi * (x_grid * 1.5 + t)
    top_r = 0.55 + 0.45 * torch.sin(phase)
    top_b = 0.55 + 0.45 * torch.sin(phase + math.pi / 2.0)
    top_g = 0.1 + 0.15 * torch.sin(phase * 0.5 - math.pi / 3.0)

    depth = torch.clamp(1.0 - (y_grid - 0.5) * 2.0, 0.0, 1.0)
    sweep = 0.5 + 0.5 * torch.sin(two_pi * (y_grid * 1.5 - t))
    bottom_b = depth
    bottom_g = sweep * (1.0 - x_grid)
    bottom_r = 0.12 + 0.18 * torch.sin(two_pi * (x_grid + t))

    r = torch.where(top_mask, top_r, bottom_r)
    g = torch.where(top_mask, top_g, bottom_g)
    b = torch.where(top_mask, top_b, bottom_b)
    a = torch.ones_like(r)

    frame = torch.stack([r, g, b, a], dim=-1)
    return frame.clamp_(0.0, 1.0).contiguous()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_grid, y_grid = build_coordinate_grid(device)

    stream_mode = os.environ.get("TORCHWINDOW_STREAM", "").lower()
    stream_enabled = stream_mode in {"1", "true", "yes", "only"}
    stream_only = stream_mode == "only"
    stream_host = os.environ.get("TORCHWINDOW_STREAM_HOST", "0.0.0.0")
    stream_port = int(os.environ.get("TORCHWINDOW_STREAM_PORT", "8765"))
    stream_codec = os.environ.get("TORCHWINDOW_STREAM_CODEC")
    stream_max_bitrate_env = os.environ.get("TORCHWINDOW_STREAM_MAX_BITRATE")
    stream_min_bitrate_env = os.environ.get("TORCHWINDOW_STREAM_MIN_BITRATE")

    def _parse_int(value: Optional[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            return None

    stream_max_bitrate = _parse_int(stream_max_bitrate_env)
    stream_min_bitrate = _parse_int(stream_min_bitrate_env)

    output_dir_env = os.environ.get("TORCHWINDOW_OUTPUT_DIR")
    save_output_dir = None if stream_only else Path(output_dir_env or "renders")

    lossyness_env = os.environ.get("TORCHWINDOW_LOSSYNESS")
    try:
        lossyness = float(lossyness_env) if lossyness_env is not None else 0.1
    except ValueError:
        lossyness = 0.1

    window = Window(
        WIDTH,
        HEIGHT,
        "Example",
        save_output_dir=save_output_dir,
        lossyness=lossyness,
        stream_enabled=stream_enabled,
        stream_only=stream_only,
        stream_host=stream_host,
        stream_port=stream_port,
        stream_max_bitrate=stream_max_bitrate,
        stream_min_bitrate=stream_min_bitrate,
        stream_max_framerate=FPS,
        stream_codec=stream_codec,
    )

    if window.output_mode == "stream":
        print(
            f"WebRTC stream ready at http://{stream_host}:{stream_port}/ — open in a WebRTC-capable browser."
        )

    frame_delay = 1.0 / FPS
    frame_budget = DURATION_SECONDS * FPS

    try:
        for frame_idx in range(frame_budget):
            frame = render_frame(x_grid, y_grid, frame_idx)
            window.draw(frame)
            if not window.running:
                break
            time.sleep(frame_delay)
    finally:
        window.close()
        if window.output_mode == "file" and save_output_dir is not None:
            print(f"Rendered fallback video written to {save_output_dir / (window.name + '.mp4')}")
