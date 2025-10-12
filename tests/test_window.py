import asyncio
import time

import aiohttp
import pytest
import torch

from torchwindow.window import Window


@pytest.mark.asyncio
async def test_window_metrics_endpoint():
    window = Window(width=64, height=64, http_port=0, prefer_hardware_encode=False)
    try:
        # give the server a moment to start
        await asyncio.sleep(0.2)
        for _ in range(3):
            tensor = torch.rand((64, 64, 3), dtype=torch.float32)
            window.draw(tensor)
            time.sleep(0.05)

        async with aiohttp.ClientSession() as session:
            url = f"http://127.0.0.1:{window.http_port}/metrics.json"
            async with session.get(url) as resp:
                assert resp.status == 200
                payload = await resp.json()
                assert payload["name"] == window.config.name
    finally:
        window.close()
