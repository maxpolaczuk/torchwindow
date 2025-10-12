import pytest

from torchwindow.config import StreamCaps, WindowConfig


def test_window_config_enforces_lossless():
    with pytest.raises(ValueError):
        WindowConfig(lossless=False)


def test_stream_caps_clamp():
    caps = StreamCaps(max_width=1920, max_height=1080, max_fps=60)
    assert caps.clamp(2560, 1440, 120) == (1920, 1080, 60)
