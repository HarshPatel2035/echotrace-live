"""Launch EchoTrace Live. Run `python main.py --help` for options."""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

from echotrace import EchoTraceAgent

__all__ = ["EchoTraceAgent", "build_agent", "parse_args"]


def fraction(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EchoTrace Live: paint with light using your webcam.")
    parser.add_argument("--camera", type=int, default=0,
                        help="webcam index (default 0; try 1, 2 ... for other cameras)")
    parser.add_argument("--width", type=int, default=1280,
                        help="requested camera width; 0 keeps the camera default (1280)")
    parser.add_argument("--height", type=int, default=720,
                        help="requested camera height; 0 keeps the camera default (720)")
    parser.add_argument("--fps", type=int, default=30,
                        help="requested camera frame rate; 0 keeps the camera default (30)")
    parser.add_argument("--work-width", type=int, default=640,
                        help="max width for effect processing; lower is faster (default 640)")
    parser.add_argument("--preset", type=int, choices=range(1, 6), metavar="1-5",
                        help="start with a visual preset")
    parser.add_argument("--no-mirror", action="store_true",
                        help="start without the horizontal mirror")
    parser.add_argument("--hud-in-captures", action="store_true",
                        help="include the on-screen HUD in snapshots and recordings")
    parser.add_argument("--flicker-limit", type=fraction, default=0.45,
                        help="ignore frames where more than this fraction of the picture "
                             "changes at once (exposure flicker); 0 disables (default 0.45)")
    parser.add_argument("--output-dir", type=Path,
                        help="folder for snapshots and recordings (default ./captures)")
    args = parser.parse_args(argv)
    if args.work_width < 160:
        parser.error("--work-width must be at least 160")
    return args


def build_agent(args: argparse.Namespace) -> EchoTraceAgent:
    agent = EchoTraceAgent(camera=args.camera, width=args.width, height=args.height,
                           fps=args.fps, work_width=args.work_width,
                           mirror=not args.no_mirror, capture_hud=args.hud_in_captures,
                           flicker_limit=args.flicker_limit, output_dir=args.output_dir)
    if args.preset:
        agent.apply_preset(args.preset)
    return agent


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(build_agent(parse_args()).run())
