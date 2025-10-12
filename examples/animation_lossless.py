import math
import time

import torch

from torchwindow import Window


def main() -> None:
    width, height = 1280, 720
    window = Window(
        width=width,
        height=height,
        name="TorchWindow 2.0 – Lossless Demo",
        stream=True,
        sfu_mode="embedded",
        http_host="0.0.0.0",
        http_port=8080,
        codec="h264",
        prefer_hardware_encode=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )

    t0 = time.time()
    try:
        while window.running:
            t = time.time() - t0
            r = torch.sqrt(xx * xx + yy * yy)
            a = torch.atan2(yy, xx) + 0.3 * t
            v = torch.sin(10.0 * r - 2.0 * t)

            r_chan = (0.5 + 0.5 * torch.sin(2 * math.pi * (a / (2 * math.pi)) + v)).clamp(0, 1)
            g_chan = (0.5 + 0.5 * torch.sin(2 * math.pi * (a / (2 * math.pi)) + 2.094 + v)).clamp(0, 1)
            b_chan = (0.5 + 0.5 * torch.sin(2 * math.pi * (a / (2 * math.pi)) + 4.188 + v)).clamp(0, 1)
            frame = torch.stack([r_chan, g_chan, b_chan], dim=-1).contiguous()

            window.draw(frame)
            time.sleep(0.0005)

            # Prevent unchecked accumulation of asynchronous CUDA work when NVENC is unavailable.
            if frame.is_cuda and window.encoder_mode != "hardware":
                torch.cuda.synchronize()
    except KeyboardInterrupt:
        pass
    finally:
        window.close()


if __name__ == "__main__":
    main()
