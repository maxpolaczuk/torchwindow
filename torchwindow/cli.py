from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from typing import Iterable, Optional

from .config import DEFAULT_STUN, WindowConfig
from .pipeline import VideoPipelineManager
from .stats import StatsCollector
from .sfu import EmbeddedSFU
from .window import Window

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torchwindow.cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchwindow",
        description="TorchWindow 2.0 - lossless WebRTC streaming for torch tensors.",
    )
    sub = parser.add_subparsers(dest="command")
    _add_serve_parser(sub)
    _add_sfu_parser(sub)
    _add_diagnose_parser(sub)
    return parser


def _add_serve_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "serve", help="Run an embedded TorchWindow server with SFU."
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--name", type=str, default="TorchWindow 2.0 (Lossless)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--codec", type=str, default="h264")
    parser.add_argument(
        "--software",
        action="store_true",
        help="Allow software fallback encoder (for development without NVENC).",
    )
    parser.add_argument(
        "--stun",
        type=str,
        nargs="*",
        default=list(DEFAULT_STUN),
        help="STUN servers (space separated).",
    )
    parser.add_argument(
        "--turn",
        type=str,
        nargs="*",
        default=[],
        help="TURN server URLs.",
    )
    parser.add_argument(
        "--caps",
        type=str,
        default=None,
        help="JSON string overriding stream caps (max_width,max_height,max_fps).",
    )
    parser.set_defaults(func=_cmd_serve)


def _add_sfu_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "sfu", help="Run a standalone SFU for remote publishing (WHIP/WHEP)."
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--name", type=str, default="TorchWindow 2.0 SFU")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--codec", type=str, default="h264")
    parser.add_argument(
        "--stun",
        type=str,
        nargs="*",
        default=list(DEFAULT_STUN),
    )
    parser.add_argument(
        "--turn",
        type=str,
        nargs="*",
        default=[],
    )
    parser.set_defaults(func=_cmd_sfu)


def _add_diagnose_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "diagnose", help="Run environment checks for TorchWindow NVENC streaming."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output.",
    )
    parser.set_defaults(func=_cmd_diagnose)


def _cmd_serve(args) -> int:
    caps = None
    if args.caps:
        try:
            caps_data = json.loads(args.caps)
            caps = (
                int(caps_data.get("max_width", 1920)),
                int(caps_data.get("max_height", 1080)),
                int(caps_data.get("max_fps", 60)),
            )
        except Exception as exc:  # pragma: no cover - CLI parsing guard
            logger.error("Failed to parse caps JSON: %s", exc)
            return 1

    window = Window(
        width=args.width,
        height=args.height,
        name=args.name,
        http_host=args.host,
        http_port=args.port,
        codec=args.codec,
        prefer_hardware_encode=not args.software,
        stun_servers=args.stun,
        turn_servers=args.turn,
    )
    if caps:
        window.set_stream_caps(*caps)

    logger.info("TorchWindow serve running at http://%s:%s", args.host, args.port)

    def _shutdown(signum, frame) -> None:  # pragma: no cover - signal handling
        logger.info("Received signal %s, shutting down.", signum)
        window.close()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while window.running:
            time.sleep(0.5)
    finally:
        window.close()
    return 0


def _cmd_sfu(args) -> int:
    config = WindowConfig(
        width=args.width,
        height=args.height,
        name=args.name,
        stream=True,
        sfu_mode="embedded",
        http_host=args.host,
        http_port=args.port,
        codec=args.codec,
        stun_servers=args.stun,
        turn_servers=args.turn,
    )

    async def runner() -> None:
        stats = StatsCollector()
        loop = asyncio.get_running_loop()
        pipeline = VideoPipelineManager(config.width, config.height, stats, loop)
        sfu = EmbeddedSFU(config, pipeline, stats, loop)
        await sfu.start()
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:  # pragma: no cover
            pass
        finally:
            await sfu.stop()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:  # pragma: no cover - manual stop
        logger.info("SFU stopped")
    return 0


def _cmd_diagnose(args) -> int:
    from .diagnostics import run_diagnostics

    report = run_diagnostics()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0 if report.get("status") == "ok" else 1


def _print_report(report: dict) -> None:
    print("TorchWindow Diagnostics")
    print("=======================")
    for section in report.get("checks", []):
        status = section.get("status", "unknown").upper()
        label = section.get("name", "Unnamed")
        print(f"[{status}] {label}")
        detail = section.get("detail")
        if detail:
            print(f"        {detail}")
    advice = report.get("advice")
    if advice:
        print("\nAdvice:")
        for line in advice:
            print(f"  - {line}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


def serve_command() -> None:
    sys.exit(main(["serve"] + sys.argv[1:]))


def sfu_command() -> None:
    sys.exit(main(["sfu"] + sys.argv[1:]))


def diagnose_command() -> None:
    sys.exit(main(["diagnose"] + sys.argv[1:]))
