"""Smoke test for both backends.

Usage:
    python -m torchwindow.example                 # auto-pick
    python -m torchwindow.example --backend cuda
    python -m torchwindow.example --backend metal
"""

import argparse
import platform
import time

import torch

from torchwindow import Window


def _default_device() -> str:
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError(
        "No CUDA or MPS device available; torchwindow needs a GPU."
    )


def _build_image(device: str) -> torch.Tensor:
    ones_v_half = torch.ones((300, 800, 1), dtype=torch.float32, device=device)
    zeros_v_half = torch.zeros((300, 800, 1), dtype=torch.float32, device=device)

    ones_h_half = torch.ones((600, 400, 1), dtype=torch.float32, device=device)
    zeros_h_half = torch.zeros((600, 400, 1), dtype=torch.float32, device=device)

    ones_whole = torch.ones((600, 800, 1), dtype=torch.float32, device=device)
    zeros_whole = torch.zeros((600, 800, 1), dtype=torch.float32, device=device)

    whole_v_half = torch.cat([ones_v_half, zeros_v_half], dim=0)
    whole_h_half = torch.cat([ones_h_half, zeros_h_half], dim=1)

    return torch.cat(
        [whole_v_half, zeros_whole, whole_h_half, ones_whole], dim=-1
    ).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["cuda", "metal"],
        default=None,
        help="Presentation stack. Defaults to metal on macOS, cuda elsewhere.",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    if args.backend == "cuda":
        device = "cuda"
    elif args.backend == "metal":
        device = "mps"
    else:
        device = _default_device()

    image = _build_image(device)
    window = Window(800, 600, "Example", backend=args.backend)

    deadline = time.time() + args.seconds
    while time.time() < deadline and getattr(window, "running", True):
        window.draw(image)
        time.sleep(1.0 / 60.0)

    window.close()


if __name__ == "__main__":
    main()
