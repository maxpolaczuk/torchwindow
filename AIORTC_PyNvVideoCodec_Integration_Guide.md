# aiortc + PyNvVideoCodec Integration Guide

This document captures the full set of steps and implementation details that were required to get TorchWindow streaming high–bitrate video from PyTorch tensors over WebRTC using **PyNvVideoCodec (NVENC)** and **aiortc**.  
Unlike the original PDF, this guide has been battle‑tested end‑to‑end and reflects the working state of the codebase after resolving payload / SDP mismatches, packetisation issues, and keyframe recovery.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [System Setup](#system-setup)  
3. [Building & Installing PyNvVideoCodec](#building--installing-pynvvideocodec)  
4. [Python Environment & TorchWindow Installation](#python-environment--torchwindow-installation)  
5. [Integration Details](#integration-details)  
   - 5.1 [Keep NVENC output in Annex‑B format](#51-keep-nvenc-output-in-annex-b-format)  
   - 5.2 [Force IDR after frame loss](#52-force-idr-after-frame-loss)  
   - 5.3 [Answer SDP handling for WHEP](#53-answer-sdp-handling-for-whep)  
   - 5.4 [Codec negotiation workflow](#54-codec-negotiation-workflow)  
   - 5.5 [Observability hooks](#55-observability-hooks)  
6. [Running the Lossless Demo](#running-the-lossless-demo)  
7. [Verification Checklist](#verification-checklist)  
8. [Troubleshooting](#troubleshooting)  
9. [Appendix: Useful Commands](#appendix-useful-commands)

---

## Prerequisites

- **GPU**: NVIDIA GPU with NVENC support (Maxwell or later). Confirm with `nvidia-smi --query-gpu=name,encoder.stats.sessionCount --format=csv`.
- **Drivers**: Latest NVIDIA Linux drivers with NVENC enabled.
- **CUDA Toolkit / NV Codec SDK**: PyNvVideoCodec needs headers from the NV Codec SDK (matching your driver).  
  Download: <https://developer.nvidia.com/nvidia-video-codec-sdk>.
- **Operating System**: Linux (Ubuntu 22.04 or later recommended).  
  The instructions assume native Linux; WSL2 works if GPU passthrough is configured.

---

## System Setup

Install build dependencies and libraries required by aiortc / PyAV:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build pkg-config \
  libffi-dev libssl-dev libjansson-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libsrtp2-dev libopus-dev libvpx-dev \
  python3-dev python3-venv python3-wheel
```

Install the GPU driver & CUDA toolkit as per NVIDIA’s documentation. Reboot after installation and verify `nvidia-smi` works.

---

## Building & Installing PyNvVideoCodec

1. **Download NV Codec SDK** and extract it somewhere accessible. Note the path to `Interface` and `Samples` directories.
2. **Clone PyNvVideoCodec** (make sure you use the fork / release that supports your SDK):

   ```bash
   git clone https://github.com/NVIDIA/PyNvVideoCodec.git
   cd PyNvVideoCodec
   ```

3. **Configure environment variables** expected by the build scripts:

   ```bash
   export NV_CODEC_SDK_ROOT=/path/to/video_codec_sdk
   export CUDA_PATH=/usr/local/cuda
   ```

4. **Build and install** into the TorchWindow virtual environment (created in the next section). Example:

   ```bash
   /path/to/torchwindow/.venv/bin/python -m pip install --upgrade pip setuptools wheel
   /path/to/torchwindow/.venv/bin/python -m pip install .
   ```

   If you maintain multiple environments, you can also publish a wheel (`python setup.py bdist_wheel`) and install it with the `nvenc` extra described below.

5. Confirm installation:

   ```bash
   /path/to/torchwindow/.venv/bin/python - <<'PY'
   import pynvvideocodec
   print("PyNvVideoCodec version:", pynvvideocodec.__version__)
   PY
   ```

---

## Python Environment & TorchWindow Installation

1. **Clone TorchWindow** (or use your working copy):

   ```bash
   git clone https://github.com/jbaron34/torchwindow.git
   cd torchwindow
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```

3. **Install TorchWindow with NVENC support** (this pulls aiortc, PyAV, NumPy, Torch, and the PyNvVideoCodec extra):

   ```bash
   pip install -e ".[nvenc]"
   ```

   If you built PyNvVideoCodec manually in the previous section, ensure that install path is visible to this environment.

---

## Integration Details

This section documents the code changes that turned the reference integration into a fully functioning pipeline. The file paths below are relative to the repository root.

### 5.1 Keep NVENC output in Annex‑B format

- **File**: `torchwindow/encoder.py` (`NvEncoder.encode`)  
- **Change**: Stop converting NVENC output into length‑prefixed NAL units (`_annexb_to_length_prefixed`).  
  aiortc’s internal H.264 packetiser expects Annex‑B. Feeding length‑prefixed data caused aiortc to transmit empty payloads or invalid RTP packets.
- **Snippet**:

  ```python
  self._ensure_sprop(bitstream)
  packet = av.Packet(bitstream)  # pass Annex-B directly
  ```

### 5.2 Force IDR after frame loss

- **File**: `torchwindow/sfu.py`, class `LosslessTrack.recv`  
- **Change**: Detect ingest gaps and immediately call `encoder.force_idr()` so the next encoded frame is a keyframe. This satisfies browser PLIs and avoids permanent decoder stalls after packet loss or frame backlog.

  ```python
  if dropped:
      self._frames_dropped += dropped
      await self._stats.record_dropped(...)
      if hasattr(self._encoder, "force_idr"):
          self._encoder.force_idr()
  ```

### 5.3 Answer SDP handling for WHEP

- **Problem**: Returning `pc.localDescription.sdp` without gathered ICE candidates produced answers with `c=0.0.0.0` and port `9`, so browsers never received media.
- **Solution**:
  1. Create the answer but patch the SDP **before** calling `setLocalDescription`.
  2. Apply the patched SDP.
  3. Wait for ICE gathering to reach `"complete"` (_with a timeout_) and then return the final `pc.localDescription.sdp`.

- **Key helper**: `_await_ice_gathering_complete` waits up to five seconds for aiortc to produce host/srflx candidates.

### 5.4 Codec negotiation workflow

To keep aiortc and the browser in sync we now:

1. Inspect the **offer** for H.264 payload IDs.
2. Reorder the aiortc transceiver’s codec list so the server sends a matching PT.
3. Patch the answer SDP to:
   - Set the correct `profile-level-id` and `sprop-parameter-sets`.
   - Filter the `m=video` line to the negotiated PT (e.g. `103/109`).

Relevant helpers:

- `_discover_h264_payloads`
- `_prioritize_h264_codecs`
- `_patch_h264_profile`
- `_filter_h264_only`

These ensure we never advertise VP8 when NVENC is actively emitting H.264.

### 5.5 Observability hooks

- Server logs include:
  - Selected payload IDs before and after patching.
  - Final gathered SDP (with real IP / port).
- Client stats (served at `/metrics.json`) show encoding time, queue depth, and per-subscriber frame drops.
- Browser console logs `framesDecoded`, `framesDropped`, `packetsReceived`, etc. Use them to confirm PLIs are satisfied.

---

## Running the Lossless Demo

1. **Launch the example** (make sure the virtual environment is active):

   ```bash
   python examples/animation_lossless.py
   ```

   You should see logs similar to:

   ```
   Streaming with chroma-sub-sampled H.264 for client compatibility (profile-level-id=42e01f).
   Prioritized H.264 payloadType=103 for mid=0
   Local SDP after gather:
   m=video 56770 UDP/TLS/RTP/SAVPF 103 104 109 114
   ...
   ```

2. **Open the browser** to <http://localhost:8080>. When the connection establishes:
   - The “Lossless Demo” canvas shows moving gradients.
   - Console logs report `framesDecoded` climbing with `framesDropped` either staying at zero or recovering (keyframe triggered).
   - `/metrics.json` lists per-client statistics.

3. Optional: capture traffic for validation.

   ```bash
   sudo tcpdump -i any -s0 udp port 56770 -w torchwindow-webrtc.pcap
   ```

   RTP packets should use PT `103` (H.264) with RTX on `104`. You’ll also see RTCP PLI / RR traffic.

---

## Verification Checklist

- [x] aiortc logs show negotiated payloads (e.g. `Prioritized H.264 payloadType=103`).
- [x] “Local SDP after gather” contains real IP/port and only the negotiated payload IDs.
- [x] Browser console stats show `framesDecoded` incrementing indefinitely; any `framesDropped` is followed by an immediate recovery.
- [x] `torchwindow-webrtc.pcap` contains SRTP packets with payload type `103`/`109` and RTCP reports.
- [x] `/metrics.json` reports the client with non-zero `encode_ms` and low `frames_dropped`.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Browser stays black, packets = 0 | Answer returned before ICE gathering | Ensure `_await_ice_gathering_complete` runs, or check firewall on port from SDP. |
| Browser stats show `framesDropped` growing rapidly, `framesDecoded` frozen | NVENC never emitted IDR | Confirm `LosslessTrack` calls `encoder.force_idr()` on drop and watch for keyframe in RTP dump. |
| aiortc still uses PT 96 (VP8) | Codec reordering failed | Check server logs for “Sender codecs after prioritization”; verify offer includes H.264 payloads. |
| Crash when installing PyNvVideoCodec | CUDA / SDK path mismatch | Verify `NV_CODEC_SDK_ROOT` and `CUDA_PATH`, rebuild wheel inside the active venv. |
| High latency or jitter | Browser/network issue | Monitor WebRTC stats (RTT, jitter), consider adjusting `fps_cap` or enabling RTX (already negotiated). |

---

## Appendix: Useful Commands

```bash
# Activate environment
source .venv/bin/activate

# Run the example (NVENC)
python examples/animation_lossless.py

# Inspect metrics
curl http://localhost:8080/metrics.json | jq

# Check negotiated SDP
tail -n +0 -f logs | grep "Local SDP"

# Packet capture for analysis
sudo tcpdump -i any -s0 udp port <video_port> -w torchwindow-webrtc.pcap

# Decode RTP in Wireshark: use profile-level-id/sprop from SDP for H.264 settings.
```

---

With these steps in place, the aiortc + PyNvVideoCodec pipeline runs reliably on TorchWindow. Use this guide as the new canonical integration reference. For further development, keep Annex‑B payloads intact, prioritise codecs dynamically, and always respond to frame loss with a forced IDR to keep WebRTC subscribers happy. Happy streaming!

