"""Render a demo video of EchoTrace Live without a webcam or a person on camera.

A synthetic scene (dim room, light wand, waving hand shape, bouncing ball) is fed
frame by frame through the real EchoTrace effect pipeline on a simulated 30 FPS clock.

    python demo/make_demo.py            # writes demo/echotrace-demo.mp4 (needs ffmpeg)
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echotrace import EchoTraceAgent  # noqa: E402

WIDTH, HEIGHT, FPS = 1280, 720, 30
OUTPUT = Path(__file__).resolve().parent / "echotrace-demo.mp4"
REPO = "github.com/HarshPatel2035/echotrace-live"
SKIN = (125, 155, 205)


class Clock:
    """Stands in for time.perf_counter so trails fade exactly as at 30 FPS."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def tick(self) -> None:
        self.now += 1 / FPS


# ---------------------------------------------------------------- scene drawing

def build_room() -> np.ndarray:
    y = np.linspace(0, 1, HEIGHT, dtype=np.float32)[:, None, None]
    room = (np.array([46, 42, 50], np.float32) * (1 - y * 0.45)).repeat(WIDTH, 1)
    room = room.astype(np.uint8)
    cv2.rectangle(room, (90, 70), (420, 360), (88, 82, 76), -1)            # window
    cv2.rectangle(room, (90, 70), (420, 360), (60, 58, 56), 6)
    cv2.line(room, (255, 70), (255, 360), (60, 58, 56), 5)
    cv2.line(room, (90, 215), (420, 215), (60, 58, 56), 5)
    cv2.rectangle(room, (930, 180), (1180, 196), (40, 52, 66), -1)         # shelf
    for i, (w, h, c) in enumerate(((34, 70, (70, 60, 110)), (28, 56, (60, 90, 70)),
                                   (40, 82, (90, 70, 60)), (30, 64, (55, 75, 105)))):
        x = 960 + i * 50
        cv2.rectangle(room, (x, 180 - h), (x + w, 180), c, -1)               # books
    cv2.rectangle(room, (0, 600), (WIDTH, HEIGHT), (30, 40, 54), -1)       # desk
    cv2.line(room, (0, 600), (WIDTH, 600), (44, 58, 74), 3)
    return room


def draw_wand(image: np.ndarray, x: float, y: float) -> None:
    center = (int(x), int(y))
    for radius, color in ((24, (100, 120, 150)), (17, (165, 195, 230)), (10, (235, 250, 255))):
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)


def draw_ball(image: np.ndarray, x: float, y: float) -> None:
    cv2.circle(image, (int(x), int(y)), 34, (40, 120, 230), -1, cv2.LINE_AA)
    cv2.circle(image, (int(x) - 10, int(y) - 12), 10, (120, 190, 255), -1, cv2.LINE_AA)


def draw_hand(image: np.ndarray, cx: float, cy: float, angle: float, scale: float = 1.0) -> None:
    """A simple open hand on a forearm, rotated about the wrist."""
    cos, sin = math.cos(angle), math.sin(angle)

    def at(dx: float, dy: float) -> tuple[int, int]:
        dx, dy = dx * scale, dy * scale
        return int(cx + dx * cos - dy * sin), int(cy + dx * sin + dy * cos)

    def limb(start, end, width) -> None:
        cv2.line(image, at(*start), at(*end), SKIN, int(width * scale), cv2.LINE_AA)

    limb((0, 0), (0, 260), 74)                                               # forearm
    cv2.ellipse(image, at(0, -58), (int(52 * scale), int(64 * scale)),
                math.degrees(angle), 0, 360, SKIN, -1, cv2.LINE_AA)          # palm
    for base_x, length, spread in ((-33, 68, -0.12), (-11, 80, -0.04),
                                   (11, 76, 0.04), (33, 58, 0.13)):
        tip = (base_x + math.sin(spread) * length, -100 - math.cos(spread) * length)
        limb((base_x, -100), tip, 22)                                        # fingers
    limb((-42, -40), (-92, -96), 24)                                         # thumb


# ---------------------------------------------------------------- text overlays

def load_font(size: int):
    try:
        from PIL import ImageFont
        return ImageFont.truetype("/System/Library/Fonts/Avenir Next.ttc", size, index=0)
    except Exception:
        return None


FONTS = {size: load_font(size) for size in (22, 30, 64)}


def put_text(image: np.ndarray, text: str, xy: tuple[int, int], size: int,
             color=(255, 255, 255), anchor: str = "la") -> np.ndarray:
    font = FONTS.get(size)
    if font is None:  # Pillow or the macOS font is unavailable: OpenCV fallback.
        scale = size / 32
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        x = xy[0] - w // 2 if anchor[0] == "m" else xy[0]
        cv2.putText(image, text, (x, xy[1] + size), cv2.FONT_HERSHEY_DUPLEX, scale,
                    color, 1, cv2.LINE_AA)
        return image
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(xy, text, font=font, fill=color[::-1], anchor=anchor)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def caption(image: np.ndarray, title: str, detail: str, key: str) -> np.ndarray:
    bar = image.copy()
    cv2.rectangle(bar, (0, HEIGHT - 92), (WIDTH, HEIGHT), (12, 10, 12), -1)
    image = cv2.addWeighted(bar, 0.62, image, 0.38, 0)
    image = put_text(image, title, (36, HEIGHT - 82), 30)
    image = put_text(image, detail, (36, HEIGHT - 42), 22, (200, 200, 205))
    if key:
        image = put_text(image, key, (WIDTH - 36, HEIGHT - 62), 22, (255, 225, 120), "ra")
    return put_text(image, "Synthetic scene · real EchoTrace effects", (WIDTH - 24, 18), 22,
                    (170, 170, 175), "ra")


def card(lines: list[tuple[str, int, tuple]], fade: float) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), (16, 10, 14), np.uint8)
    y = HEIGHT // 2 - 30 * len(lines)
    for text, size, color in lines:
        image = put_text(image, text, (WIDTH // 2, y), size, color, "ma")
        y += int(size * 1.6)
    return (image.astype(np.float32) * fade).astype(np.uint8)


# ---------------------------------------------------------------- motion paths

def wave(t: float, speed: float = 1.0) -> tuple[float, float, float]:
    return 640 + 200 * math.sin(t * 1.1 * speed), 470, 0.45 * math.sin(t * 4.5 * speed)


def figure_eight(t: float) -> tuple[float, float]:
    a = t * 1.7
    d = 1 + math.sin(a) ** 2
    return 640 + 420 * math.cos(a) / d, 300 + 420 * math.sin(a) * math.cos(a) / d


def spiral(t: float) -> tuple[float, float]:
    r = 40 + 55 * t
    return 640 + r * math.cos(t * 3.2), 310 + r * 0.75 * math.sin(t * 3.2)


def bounce(t: float) -> tuple[float, float]:
    x = 180 + (t * 260) % 920
    return x, 560 - abs(math.sin(t * 3.4)) * 380


def kick(t: float) -> np.ndarray:
    """2048 microphone samples: a bass kick every 0.9 s plus a little hi-hat."""
    samples = np.arange(2048) / 48000
    since = t % 0.9
    bass = 0.35 * math.exp(-since * 9) * np.sin(2 * np.pi * 70 * samples)
    hat = 0.02 * np.sin(2 * np.pi * 5200 * samples) * (since < 0.1)
    return (bass + hat).astype(np.float32)


# ---------------------------------------------------------------- the demo

def scenes():
    """(seconds, setup, draw(frame, t), caption) for each part of the demo."""
    def mode(name):
        return lambda agent: setattr(agent, "mode_index", agent.MODES.index(name))

    def preset(number):
        return lambda agent: agent.apply_preset(number)

    def audio(agent):
        agent.mode_index = 0
        agent.audio_enabled = True

    yield (5, mode("Echo"), lambda f, t: draw_hand(f, *wave(t)),
           ("Echo", "Moving things leave glowing trails that slowly fade", "C: change mode"))
    yield (6, mode("Neon"), lambda f, t: draw_hand(f, *wave(t, 0.45 if t < 3 else 1.6)),
           ("Neon", "Slow motion glows cyan, fast motion glows magenta", "C: change mode"))
    yield (5, mode("Ghost"), lambda f, t: draw_hand(f, *wave(t)),
           ("Ghost", "Translucent echoes of the moving image", "C: change mode"))
    yield (5, mode("Fire"), lambda f, t: draw_ball(f, *bounce(t)),
           ("Fire", "Hot, fire-coloured heat traces", "C: change mode"))
    yield (5, mode("Rainbow"), lambda f, t: draw_wand(f, *spiral(t)),
           ("Rainbow", "Shifting rainbow hues that linger on the canvas", "C: change mode"))
    yield (6, preset(3), lambda f, t: draw_wand(f, *figure_eight(t)),
           ("Rainbow Mandala", "Six-way symmetry with particles", "Preset 3"))
    yield (5, preset(5), lambda f, t: (draw_ball(f, *bounce(t)), draw_hand(f, *wave(t))),
           ("Fire Vortex", "Four-way symmetry, particles and strong glow", "Preset 5"))
    yield (6, preset(4), lambda f, t: draw_hand(f, *wave(t, 0.8)),
           ("Ghost Time", "Slit-scan: a sweeping slice stretches motion through time",
            "Preset 4"))
    yield (5, audio, lambda f, t: draw_wand(f, *figure_eight(t)),
           ("Audio reactive", "Bass beats send out rings and boost the glow (test beat)",
            "A: microphone"))


def main() -> int:
    room = build_room()
    rng = np.random.default_rng(7)
    clock = Clock()
    noise = np.empty((HEIGHT, WIDTH, 3), np.int16)
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
               "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(OUTPUT)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    write = lambda image: encoder.stdin.write(np.ascontiguousarray(image).tobytes())

    title = [("EchoTrace Live", 64, (255, 235, 120)),
             ("Paint with light using your webcam", 30, (235, 235, 240)),
             ("Runs locally on your CPU · Python, OpenCV, MediaPipe", 22, (170, 170, 180))]
    for i in range(FPS * 3):
        write(card(title, min(1.0, i / 12, (FPS * 3 - i) / 8)))

    with patch("time.perf_counter", clock):
        agent = EchoTraceAgent(mirror=False)
        agent.rng = np.random.default_rng(3)
        for seconds, setup, draw, text in scenes():
            agent.close_audio()
            agent.particles_enabled = agent.time_slice = False
            agent.kaleidoscope = 0
            agent.decay, agent.glow = 0.96, 1.4
            setup(agent)
            agent.reset_canvas()
            print(f"Rendering: {text[0]}")
            for i in range(FPS * seconds):
                t = i / FPS
                frame = room.copy()
                draw(frame, t)
                # Gentle sensor noise, like a real webcam, below the motion threshold.
                cv2.randn(noise, 0, 2)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                if agent.audio_enabled:
                    agent.audio_blocks.append((clock(), kick(t)))
                output = caption(agent.process_frame(frame), *text)
                fade = min(1.0, i / 6, (FPS * seconds - i) / 6)
                write((output.astype(np.float32) * fade).astype(np.uint8))
                clock.tick()
        agent.close_audio()

    outro = [("EchoTrace Live", 64, (255, 235, 120)),
             (REPO, 30, (235, 235, 240)),
             ("pip install -r requirements.txt   ·   python main.py", 22, (170, 170, 180))]
    for i in range(FPS * 4):
        write(card(outro, min(1.0, i / 12, (FPS * 4 - i) / 12)))

    encoder.stdin.close()
    if encoder.wait() != 0:
        print("ffmpeg failed; is it installed?")
        return 1
    print(f"Saved {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
