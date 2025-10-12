import torch

from av.video.frame import VideoFrame

from torchwindow.encoder import create_encoder
from torchwindow.pipeline import ViewportState


def test_software_encoder_returns_video_frame():
    encoder = create_encoder(
        width=32,
        height=32,
        codec="h264",
        fps=30,
        prefer_hardware=False,
    )
    frame = torch.zeros((32, 32, 3), dtype=torch.float32)
    viewport = ViewportState(target_size=(32, 32), fps_cap=30)
    packet = encoder.encode(frame, viewport)
    assert isinstance(packet, VideoFrame)
