import asyncio

import pytest
import torch

from torchwindow.pipeline import ViewportState, VideoPipelineManager
from torchwindow.stats import StatsCollector


@pytest.mark.asyncio
async def test_pipeline_compose_zoom_and_resize():
    stats = StatsCollector()
    loop = asyncio.get_running_loop()
    pipeline = VideoPipelineManager(64, 64, stats, loop)

    frame = torch.linspace(0, 1, 64 * 64 * 3).reshape(64, 64, 3)
    await pipeline.submit_frame(frame)
    record = await pipeline.next_frame(0)

    viewport = ViewportState(
        view_id="default",
        center_px=(32, 32),
        zoom=2.0,
        target_size=(32, 32),
        fps_cap=30,
    )
    composed = await pipeline.compose_view(record, viewport)
    assert composed.shape == (32, 32, 3)
    assert torch.isclose(composed.float().mean(), torch.tensor(0.5), atol=0.5)
