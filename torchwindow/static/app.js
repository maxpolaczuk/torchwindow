const videoEl = document.getElementById("viewport");
const statusEl = document.getElementById("connection-status");
const nameEl = document.getElementById("window-name");
const resolutionSelect = document.getElementById("resolution-select");
const statServerFps = document.getElementById("stat-server-fps");
const statEncode = document.getElementById("stat-encode");
const statEncoder = document.getElementById("stat-encoder");
const statQueue = document.getElementById("stat-queue");
const statDropped = document.getElementById("stat-dropped");
const statSubscribers = document.getElementById("stat-subscribers");

let pc;
let controlChannel;
let zoomLevel = 1.0;
let center = { x: 0, y: 0 };
let caps = { w: 1280, h: 720, fps: 60 };

async function start() {
  await refreshMetrics();
  await connect();
  setInterval(refreshMetrics, 1000);
}

async function connect() {
  statusEl.textContent = "Negotiating…";
  pc = new RTCPeerConnection();
  pc.addEventListener("track", (event) => {
    videoEl.srcObject = event.streams[0];
    console.log("[TorchWindow] Received track:", {
      kind: event.track.kind,
      id: event.track.id,
      streamId: event.streams[0]?.id,
    });
  });

  pc.addEventListener("connectionstatechange", () => {
    statusEl.textContent = pc.connectionState;
    console.log("[TorchWindow] Peer connection state:", pc.connectionState);
  });

  pc.addTransceiver("video", { direction: "recvonly" });

  controlChannel = pc.createDataChannel("control");
  controlChannel.onopen = () => {
    statusEl.textContent = "connected";
    console.log("[TorchWindow] Control channel open");
  };

  controlChannel.onclose = () => {
    statusEl.textContent = "channel closed";
    console.warn("[TorchWindow] Control channel closed");
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  console.log("[TorchWindow] Local SDP offer created", offer.sdp);
  const response = await fetch("/whep", {
    method: "POST",
    headers: { "Content-Type": "application/sdp" },
    body: offer.sdp,
  });

  if (!response.ok) {
    statusEl.textContent = `error: ${response.status}`;
    console.error("[TorchWindow] WHEP negotiation failed", response.status);
    throw new Error("Failed to negotiate session");
  }

  const answer = await response.text();
  console.log("[TorchWindow] Remote SDP answer received", answer);
  await pc.setRemoteDescription({ type: "answer", sdp: answer });
  statusEl.textContent = "connected";
  console.log("[TorchWindow] Remote description set");
}

async function refreshMetrics() {
  try {
    const response = await fetch("/metrics.json");
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    nameEl.textContent = data.name;
    statServerFps.textContent = data.server_fps.toFixed(1);
    statEncode.textContent = data.encode_ms_avg.toFixed(2);
    const mode = data.encoder_mode || "unknown";
    statEncoder.textContent = data.encoder_detail
      ? `${mode} (${data.encoder_detail})`
      : mode;
    statQueue.textContent = data.queue_depth;
    statDropped.textContent = data.dropped_frames;
    statSubscribers.textContent = JSON.stringify(data.per_client, null, 2);
  } catch (error) {
    console.error("Failed to fetch metrics", error);
  }

  if (pc) {
    const stats = await pc.getStats(null);
    const report = Array.from(stats.values());
    const inboundVideo = report.find((item) => item.type === "inbound-rtp" && item.kind === "video");
    if (inboundVideo) {
      console.log("[TorchWindow] Inbound RTP stats", {
        framesDecoded: inboundVideo.framesDecoded,
        framesDropped: inboundVideo.framesDropped,
        bytesReceived: inboundVideo.bytesReceived,
        packetsReceived: inboundVideo.packetsReceived,
      });
    }
    let candidate;
    stats.forEach((report) => {
      if (
        report.type === "candidate-pair" &&
        report.state === "succeeded" &&
        report.currentRoundTripTime
      ) {
        candidate = report;
      }
    });
    if (candidate && controlChannel?.readyState === "open") {
      controlChannel.send(
        JSON.stringify({
          cmd: "metrics",
          rtt: candidate.currentRoundTripTime * 1000,
        }),
      );
    }
  }
}

function sendCommand(payload) {
  if (!controlChannel || controlChannel.readyState !== "open") {
    console.warn("Control channel unavailable");
    return;
  }
  controlChannel.send(JSON.stringify(payload));
}

document.querySelectorAll("button[data-action]").forEach((button) => {
  button.addEventListener("click", () => {
    const action = button.dataset.action;
    switch (action) {
      case "pause":
        sendCommand({ cmd: "pause" });
        break;
      case "resume":
        sendCommand({ cmd: "resume" });
        break;
      case "zoom-in":
        zoomLevel = Math.min(zoomLevel + 0.2, 5.0);
        sendCommand({ cmd: "zoom", scale: zoomLevel, center: [center.x, center.y] });
        break;
      case "zoom-out":
        zoomLevel = Math.max(1.0, zoomLevel - 0.2);
        sendCommand({ cmd: "zoom", scale: zoomLevel, center: [center.x, center.y] });
        break;
      case "pan-up":
        center.y -= 40;
        sendCommand({ cmd: "pan", dx: 0, dy: -40 });
        break;
      case "pan-down":
        center.y += 40;
        sendCommand({ cmd: "pan", dx: 0, dy: 40 });
        break;
      case "pan-left":
        center.x -= 40;
        sendCommand({ cmd: "pan", dx: -40, dy: 0 });
        break;
      case "pan-right":
        center.x += 40;
        sendCommand({ cmd: "pan", dx: 40, dy: 0 });
        break;
    }
  });
});

resolutionSelect.addEventListener("change", () => {
  const [wh, fps] = resolutionSelect.value.split("@");
  const [w, h] = wh.split("x").map((v) => parseInt(v, 10));
  const fpsValue = parseInt(fps, 10);
  caps = { w, h, fps: fpsValue };
  sendCommand({ cmd: "resize", w, h, fps: fpsValue });
});

start().catch((error) => {
  console.error("TorchWindow boot failed", error);
  statusEl.textContent = "error";
});
