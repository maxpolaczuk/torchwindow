Below is a clean, hand-offable plan for **TorchWindow 2.0** with **lossless-only WebRTC streaming**. It defaults to a **co-hosted SFU** (pure Python, aiortc-based) and can also run that **same SFU remotely**. It uses **aiohttp** to serve a minimal `index.html` that auto-connects, shows the **window name**, and live **stats** (FPS, encode time, RTT, dropped frames). For encoding, it uses **PyNvVideoCodec** (NVENC) configured for **true lossless**. No other streaming formats, no fallbacks, no websockets.

---

# 0) Goals & Non-Goals

**Goals**

* Headless, zero-OpenGL path; runs on B200-class servers.
* **Lossless-only** end-to-end video: perfect reconstruction at the viewer.
* Default “just works”: single process, co-hosted SFU, one command.
* Pure **WebRTC** for media + **DataChannel** for controls.
* Server-side zoom/pan (render true viewports, not digital zoom).
* Same SFU can run remotely from the app with one switch.
* Minimal dependencies: **aiohttp**, **aiortc**, **PyNvVideoCodec**, **torch** (or cupy), optional cpu libs.

**Non-Goals**

* No lossy encoding.
* No simulcast/SVC, no multi-codec juggling, no websockets.

> **Browser support note:** We target **Chromium-based browsers** with AV1/HEVC lossless decode capability. (Lossless H.264/HEVC/AV1 is constrained by browser implementations; to keep things simple and truly lossless, we document supported browsers explicitly.)

---

# 1) Public API (Python)

```python
from torchwindow import Window

# Simple: embedded SFU, serves at http://0.0.0.0:8080/
win = Window(
    width=1280, height=720,
    name="TorchWindow 2.0 (Lossless)",
    stream=True,
    sfu_mode="embedded",            # "embedded" | "remote"
    http_host="0.0.0.0",
    http_port=8080,
    stun_servers=["stun:stun.l.google.com:19302"],
    turn_servers=[],                 # optional TURNs
    lossless=True,                   # enforced; only mode supported
    codec="av1",                     # "av1" preferred, fallback "hevc" if specified and supported
    prefer_hardware_encode=True      # use PyNvVideoCodec (NVENC) if available
)

# Optional per-client caps (lossless always true)
win.set_stream_caps(max_width=1920, max_height=1080, max_fps=60)
```

**Remote SFU mode** (same library, different process/host):

```python
win = Window(
    width=1280, height=720, name="TorchWindow 2.0 (Lossless)",
    stream=True, sfu_mode="remote",
    sfu_url="https://sfu.example.com",  # launched via `torchwindow sfu`
    stun_servers=["stun:stun.l.google.com:19302"],
    turn_servers=[],
    lossless=True, codec="av1", prefer_hardware_encode=True
)
```

**CLI helpers (packaged entry points):**

```
torchwindow serve         # app + embedded SFU
torchwindow sfu --port 8443 --public-host sfu.example.com  # SFU only (remote)
```

---

# 2) Process Model & Components

**Single process (default):**

* `AiohttpApp` – serves `index.html`, `/app.js`, `/app.css`, `/metrics.json`.
* `EmbeddedSFU` – pure-Python SFU (aiortc):

  * **WHIP** ingest (internal publisher = app).
  * **WHEP** egress (browsers).
  * **RtpRouter** – forwards RTP losslessly; no transcoding.
  * **DataChannelHub** – pause/resume, zoom/pan, resize.
* `VideoPipelineManager`:

  * **FrameIngest** – lock-free ring (depth=2).
  * **ViewportComposer** – server-side crop/scale **on GPU** (torch.cuda/cupy); CPU fallback.
  * **Encoder** – **PyNvVideoCodec** (NVENC) lossless; software lossless only if explicitly enabled.
  * **Packetizer** → **EmbeddedSFU.RtpRouter**.
* `StatsCollector` – FPS, encode ms, queue depth, RTP out, RTT, dropped frames.

**Remote SFU mode:**

* App runs `AiohttpApp` + `VideoPipelineManager`, **publishes via WHIP** to remote `torchwindow sfu`.
* Browsers connect to remote SFU via **WHEP**.

---

# 3) Lossless-Only Encoding (PyNvVideoCodec)

**Principles**

* **Perfect reconstruction**: no chroma subsampling loss, no quantization.
* Use **PyNvVideoCodec** to drive NVENC in **lossless mode**.
* Preferred codec: **AV1 lossless** (where HW encode + browser decode is available).
* Alternative (if explicitly set): **HEVC lossless** (browser support varies; document).

**Input pixel pipeline**

* Simulation frame tensor: GPU `float32[0..1]` or `uint8` RGB(A).
* GPU convert to **YUV 4:4:4** (or native RGB if codec supports RGB lossless) to avoid chroma loss.
* Feed device pointer to PyNvVideoCodec.

**NVENC config (illustrative)**

* `codec="av1"` (or `"hevc"`)
* `lossless=True` (maps to QP=0 + lossless tools, 4:4:4 where required)
* `gop=1` or intra as desired (lossless still benefits from inter; keep simple with small GOP)
* `b_frames=0`
* `rc_mode="constqp"` with `qp=0` or vendor lossless flag
* `preset="hq"` (latency remains low; no lossy tuning params used)

> **Note:** We explicitly do **not** support lossy modes or subsampled 4:2:0 in 2.0.

---

# 4) Server-Side Viewports (Zoom/Pan)

**Per-subscriber viewport state**

```python
Viewport {
  view_id: str,                    # which source (e.g., "global" / "camera:1")
  center_px: tuple[int, int],      # pixel center in source frame/world-projected
  zoom: float,                     # 1.0 = full; >1 zoom-in
  target_size: (w, h),             # negotiated; must be <= max caps
  fps_cap: int                     # <= max_fps
}
```

**ViewportComposer**

* **GPU path**: batched crops/scales via torch CUDA (or cupy) kernels.
* **CPU fallback**: numpy (optionally cv2) if CUDA not present.
* If multiple subscribers share the same viewport & size, render once and fan out.
* Always produce **lossless input surfaces** to encoder.

**Controls via DataChannel**

```json
{"cmd":"pause"} | {"cmd":"resume"}
{"cmd":"zoom","center":[x,y],"scale":1.5}
{"cmd":"pan","dx":10,"dy":-5}
{"cmd":"resize","w":1280,"h":720,"fps":30}
{"cmd":"select_view","id":"global"}
```

---

# 5) HTTP & WebRTC Endpoints (aiohttp)

* `GET /` → `index.html` (vanilla HTML + inline ES modules; no build step)
* `GET /app.js`, `GET /app.css` → minimal UI/logic
* `GET /metrics.json` → server stats (window name, fps, encode ms, rtt, drops, subscribers)
* `POST /whip` → SDP offer for publisher (internal or remote app)
* `POST /whep` → SDP offer for browser subscribers

**Browser (`/app.js`)**

* Create `RTCPeerConnection`, POST SDP to `/whep`, set remote.
* Attach video to `<video>` element.
* Open `DataChannel` for controls (pause/zoom/pan/resize).
* Poll `/metrics.json` (1s) and combine with `pc.getStats()` to show FPS/RTT/drops.

---

# 6) Embedded SFU (aiortc)

* **RtpRouter** forwards publisher SSRC to N subscribers **unchanged**.
* No transcoding, no simulcast/SVC; **one lossless stream per subscribed viewport**.
* Handle RTCP PLI/FIR (request keyframe on viewport/size jumps).
* ICE: inject configured STUN/TURN into peers.
* Identical code runs as **remote SFU** via `torchwindow sfu`.

---

# 7) Backpressure & Latency (kept simple)

* **Ring buffer depth=2** per stream (drop oldest on overflow).
* Encoders run async; NVENC via PyNvVideoCodec.
* Keep minimal pipeline delay; intraframe or small GOP OK (still lossless).
* Stats: `server_fps`, `encode_ms_avg`, `queue_depth`, `dropped_frames`, `subscribers`, and per-client RTT.

---

# 8) Example App: Colorful Procedural Animation (for confirmation)

A self-contained example producing a vibrant, time-varying, **lossless** animation. It renders entirely on GPU when available:

```python
# examples/animation_lossless.py
import math, time, torch
from torchwindow import Window

W, H = 1280, 720
win = Window(
    width=W, height=H,
    name="TorchWindow 2.0 – Lossless Demo",
    stream=True, sfu_mode="embedded",
    http_host="0.0.0.0", http_port=8080,
    stun_servers=["stun:stun.l.google.com:19302"],
    lossless=True, codec="av1", prefer_hardware_encode=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yy, xx = torch.meshgrid(torch.linspace(-1,1,H,device=device),
                        torch.linspace(-1,1,W,device=device), indexing="ij")

t0 = time.time()
while win.running:
    t = time.time() - t0
    # Simple colorful function: concentric waves + rotation in HSV-like space
    r = torch.sqrt(xx*xx + yy*yy)
    a = torch.atan2(yy, xx) + 0.3*t
    v = torch.sin(10.0*r - 2.0*t)

    # Map to RGB in [0,1]
    R = (0.5 + 0.5*torch.sin(6.283*(a/ (2*math.pi)) + 0.0 + v)).clamp(0,1)
    G = (0.5 + 0.5*torch.sin(6.283*(a/ (2*math.pi)) + 2.094 + v)).clamp(0,1)
    B = (0.5 + 0.5*torch.sin(6.283*(a/ (2*math.pi)) + 4.188 + v)).clamp(0,1)

    frame = torch.stack([R, G, B], dim=-1).to(device)          # HxWx3 float32 [0,1]
    win.draw(frame)                                            # non-blocking
```

* Run: `python examples/animation_lossless.py`
* Open: `http://<server>:8080/`
  You’ll see a crisp, **perfectly reconstructed**, colorful animation; UI shows window name + live stats.

---

# 9) Stats & UI

**`GET /metrics.json`** (example):

```json
{
  "name":"TorchWindow 2.0 – Lossless Demo",
  "server_fps": 59.8,
  "encode_ms_avg": 3.4,
  "queue_depth": 0,
  "dropped_frames": 0,
  "subscribers": 3,
  "per_client":[
    {"id":"c1","w":1280,"h":720,"fps":60,"rtt_ms":18,"frames_dropped":0}
  ]
}
```

**`index.html`**

* Displays **window name**, `<video>` (autoplay/playsinline/muted).
* Buttons: Pause / Resume, Zoom ±, Pan arrows, Set Size (720p/1080p).
* Live stats panel from `/metrics.json` + `getStats()`.

---

# 10) Configuration & Defaults

* `http_host="0.0.0.0"`, `http_port=8080`.
* `lossless=True` is **mandatory** (the only mode).
* `codec="av1"` (preferred); alternative `"hevc"` if you know your clients can losslessly decode it.
* ICE: default STUN as shown; TURN optional.
* TLS: front with a reverse proxy if needed; aiohttp can serve HTTP for LAN/dev.

**Env overrides**

```
TORCHWINDOW_HTTP_PORT
TORCHWINDOW_STUN
TORCHWINDOW_TURN
TORCHWINDOW_PUBLIC_URL
TORCHWINDOW_CODEC   # "av1"|"hevc"
```

---

# 11) Dependency Footprint

* **Required:** `aiohttp`, `aiortc`, `torch` (or `cupy`), `numpy`
* **Encoding:** **PyNvVideoCodec** (NVENC) – packaged as `torchwindow[nvenc]`
* **Optional:** `opencv-python` (CPU scale/crop fallback)

---

# 12) Deliverables

* Python package `torchwindow` with extra `[nvenc]` for PyNvVideoCodec.
* CLIs: `torchwindow serve`, `torchwindow sfu`.
* `static/` assets: `index.html`, `app.js`, `app.css` (tiny, no build).
* Example: `examples/animation_lossless.py`.
* Docs: quickstart (embedded SFU), remote SFU guide, browser compatibility for **lossless**, TURN tips.
* Tests: endpoints, SFU routing, lossless encode path (PyNvVideoCodec), controls, metrics.

---

## Notes for the developer

* Treat **lossless** as a hard invariant: enforce 4:4:4 (or RGB) paths; reject configs that imply subsampling or QP>0.
* Prefer **AV1 lossless** on capable NVENC + Chromium clients; otherwise allow `"hevc"` only if explicitly requested and verified for clients.
* Keep the pipeline lean: one viewport → one lossless encoder instance → SFU forwarder; **no** simulcast, **no** multi-codec.
* Maintain backpressure by dropping the oldest frame (never block the simulation).
* Keep `index.html` dead simple; everything should **“just work”** with defaults.
