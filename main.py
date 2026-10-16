"""EchoTrace Live: entirely local, CPU-friendly webcam light painting."""

from __future__ import annotations

import math
import importlib.util
import multiprocessing
import platform
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


_worker_tracker = None


def detect_hands_local(model_path: str, rgb: np.ndarray, timestamp: int):
    """Isolated local worker: even a native MediaPipe abort cannot kill the UI."""
    global _worker_tracker
    import mediapipe as mp
    if _worker_tracker is None:
        from multiprocessing.util import Finalize
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path,
                                             delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.6, min_tracking_confidence=0.6)
        _worker_tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        Finalize(None, _worker_tracker.close, exitpriority=10)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _worker_tracker.detect_for_video(image, timestamp)
    # Serialize only landmarks/categories, never native handles or camera data.
    return SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=p.x, y=p.y, z=p.z) for p in hand]
                        for hand in result.hand_landmarks],
        handedness=[[SimpleNamespace(category_name=c.category_name) for c in categories]
                    for categories in result.handedness])


class EchoTraceAgent:
    WINDOW = "EchoTrace Live"
    MODES = ("Echo", "Neon", "Ghost", "Fire", "Rainbow")

    def __init__(self) -> None:
        self.capture = None
        self.writer = None
        self.record_path = None
        self.record_size = None
        self.record_started = 0.0
        self.record_frames = 0
        self.record_fps = 30.0
        self.output_dir = Path(__file__).resolve().parent / "captures"
        self.mode_index = 0
        # Retention per 1/30 second: larger numbers mean longer trails.
        self.decay = 0.96
        self.glow = 1.4
        self.mirror = True
        self.paused = False
        self.show_help = True
        self.running = True
        self.canvas = None
        self.previous_gray = None
        self.vignette = None
        self.frame_shape = None
        self.latest_result = None
        self.fps = 30.0
        self.last_process_time = None
        self.notice = "Move your hands to paint with light"
        self.notice_until = time.perf_counter() + 5.0
        self.show_hud = True
        self.frozen = False
        self.gesture_frozen = False
        self.effect_frame = None
        self.gestures_enabled = False
        self.hand_executor = None
        self.hand_future = None
        self.hand_model_path = None
        self.hand_submit_time = 0.0
        self.hand_result_time = 0.0
        self.hand_timestamp = 0
        self.gesture_candidate = ""
        self.gesture_count = 0
        self.gesture_latched = ""
        self.gesture_action_time = 0.0
        self.two_hands = False
        self.finger_points = {}
        self.stroke_layer = None
        self.audio_enabled = False
        self.audio_stream = None
        self.audio_blocks = deque(maxlen=4)
        self.audio_rate = 48000
        self.audio_status = ""
        self.bass = self.treble = 0.0
        self.bass_average = 0.001
        self.beat_time = 0.0
        self.rings = []
        self.particles_enabled = False
        # x, y, vx, vy, remaining life, initial life, B, G, R
        self.particles = np.empty((0, 9), np.float32)
        self.spawn_credit = 0.0
        self.rng = np.random.default_rng()
        self.kaleidoscope = 0
        self.kaleidoscope_maps = {}
        self.time_slice = False
        self.slice_canvas = None
        self.slice_position = 0.0
        self.effective_glow = self.glow

    @property
    def mode(self) -> str:
        return self.MODES[self.mode_index]

    def notify(self, message: str) -> None:
        print(message)
        self.notice = message
        self.notice_until = time.perf_counter() + 4.0

    def open_camera(self) -> bool:
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened() and platform.system() == "Windows":
            self.capture.release()
            self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            print("ERROR: Could not open webcam 0. Check camera permissions and close "
                  "other camera apps. See README.md for troubleshooting.")
            return False
        # These are requests; use the actual frame size returned by the camera.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def reset_canvas(self) -> None:
        if self.canvas is not None:
            self.canvas.fill(0)
        self.previous_gray = None
        self.last_process_time = None
        self.particles = np.empty((0, 9), np.float32)
        self.rings.clear()
        self.finger_points.clear()
        if self.stroke_layer is not None:
            self.stroke_layer.fill(0)
        self.slice_canvas = None
        self.slice_position = 0.0

    def prepare_buffers(self, frame: np.ndarray) -> tuple[int, int]:
        height, width = frame.shape[:2]
        # Motion, persistent float canvas, and blur run at <= 640 pixels wide.
        scale = min(1.0, 640.0 / width)
        work_width = max(1, round(width * scale))
        work_height = max(1, round(height * scale))
        if self.frame_shape != (height, width):
            if self.writer is not None:
                self.stop_recording()
                self.notify("Camera size changed; recording stopped. Press V to restart.")
            self.frame_shape = (height, width)
            self.canvas = np.zeros((work_height, work_width, 3), np.float32)
            self.stroke_layer = np.zeros_like(self.canvas)
            self.kaleidoscope_maps.clear()
            self.reset_canvas()
            self.effect_frame = None
            self.previous_gray = None
            y = np.linspace(-1, 1, work_height, dtype=np.float32)[:, None]
            x = np.linspace(-1, 1, work_width, dtype=np.float32)[None, :]
            self.vignette = np.clip(1.0 - 0.18 * (x * x + y * y), 0.65, 1)[..., None]
        return work_width, work_height

    def apply_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.previous_gray is None:
            mask = np.zeros(gray.shape, np.uint8)
        else:
            difference = cv2.absdiff(gray, self.previous_gray)
            _, mask = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.dilate(mask, None, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
        self.previous_gray = gray
        return mask

    @staticmethod
    def hue_color(hue: float) -> np.ndarray:
        pixel = np.array([[[int(hue) % 180, 235, 255]]], np.uint8)
        return cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)

    def colorize(self, frame: np.ndarray, mask: np.ndarray,
                 edges: np.ndarray, now: float) -> np.ndarray:
        strength = mask.astype(np.float32)[..., None] / 255.0
        if self.mode == "Neon":
            # Motion speed/intensity interpolates between cyan and magenta.
            cyan = np.array([255, 255, 20], np.float32)
            magenta = np.array([255, 20, 255], np.float32)
            color = cyan * (1 - strength) + magenta * strength
            edge_strength = edges.astype(np.float32)[..., None] / 255.0
            return color * strength + edge_strength * strength * cyan * 0.6
        if self.mode == "Ghost":
            tint = frame.astype(np.float32) * 0.55 + np.array([110, 65, 75], np.float32)
            return tint * strength * 0.8
        if self.mode == "Fire":
            heat = np.clip(mask.astype(np.float32) * 0.75 + edges * 0.25, 0, 255).astype(np.uint8)
            return cv2.applyColorMap(heat, cv2.COLORMAP_HOT).astype(np.float32) * strength
        if self.mode == "Rainbow":
            height, width = mask.shape
            hues = (np.arange(width, dtype=np.float32)[None, :] * 100 / max(1, width)
                    + np.arange(height, dtype=np.float32)[:, None] * 35 / max(1, height)
                    + now * 28) % 180
            hsv = np.empty_like(frame)
            hsv[..., 0] = hues.astype(np.uint8)
            hsv[..., 1] = 235
            hsv[..., 2] = 255
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) * strength
        return strength * self.hue_color(now * 16)

    def apply_glow(self, trace: np.ndarray) -> np.ndarray:
        glow = self.effective_glow
        if glow <= 0:
            return trace
        blur = cv2.GaussianBlur(trace, (0, 0), sigmaX=2 + glow * 2)
        return np.clip(trace + blur * glow, 0, 255)

    def toggle_gestures(self) -> None:
        if self.gestures_enabled:
            self.close_gestures()
            self.notify("Gestures OFF")
            return
        try:
            if importlib.util.find_spec("mediapipe") is None:
                raise ImportError("Install mediapipe from requirements.txt")
            model = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
            if not model.is_file():
                raise FileNotFoundError("Missing models/hand_landmarker.task; see README")
            self.hand_model_path = str(model)
            self.hand_executor = ProcessPoolExecutor(max_workers=1,
                                                     mp_context=multiprocessing.get_context("spawn"))
            self.gestures_enabled = True
            self.notify("Gestures ON: initializing local hand tracker")
        except Exception as error:
            self.close_gestures()
            self.notify(f"Gestures unavailable: {error}")

    def close_gestures(self) -> None:
        self.gestures_enabled = False
        if self.hand_executor is not None:
            # Finish the single in-flight job before closing its native tracker.
            self.hand_executor.shutdown(wait=True, cancel_futures=True)
            self.hand_executor = None
        self.hand_future = None
        self.gesture_frozen = self.two_hands = False
        self.gesture_candidate = self.gesture_latched = ""
        self.gesture_count = 0
        self.finger_points.clear()

    @staticmethod
    def classify_hand(landmarks) -> tuple[str, np.ndarray]:
        points = np.array([(p.x, p.y, p.z) for p in landmarks], np.float32)
        palm = max(float(np.linalg.norm(points[0] - points[9])), 0.01)
        extended = [np.linalg.norm(points[tip] - points[0]) >
                    np.linalg.norm(points[pip] - points[0]) * 1.18
                    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))]
        pinch = np.linalg.norm(points[4] - points[8]) / palm < 0.32
        if pinch:
            return "pinch", points
        if all(extended):
            return "palm", points
        if not any(extended):
            return "fist", points
        if extended[0] and not any(extended[1:]):
            return "index", points
        return "neutral", points

    def consume_hands(self, result, now: float) -> None:
        hands = result.hand_landmarks
        self.two_hands = len(hands) >= 2
        observations = []
        next_points = {}
        height, width = self.canvas.shape[:2]
        for index, landmarks in enumerate(hands):
            gesture, points = self.classify_hand(landmarks)
            observations.append(gesture)
            if gesture == "index":
                categories = result.handedness[index] if index < len(result.handedness) else []
                identity = categories[0].category_name if categories else str(index)
                tip = tuple(np.clip(points[8, :2] * [width - 1, height - 1],
                                    [0, 0], [width - 1, height - 1]).astype(int))
                previous = self.finger_points.get(identity)
                color = tuple(float(c) for c in self.hue_color(now * 30))
                if previous is not None and math.dist(previous, tip) < width * 0.35:
                    cv2.line(self.stroke_layer, previous, tip, color, 3, cv2.LINE_AA)
                else:
                    cv2.circle(self.stroke_layer, tip, 3, color, -1)
                next_points[identity] = tip
        self.finger_points = next_points
        # Discrete actions require two consecutive classifications. Holding a pose
        # triggers once; release it before repeating. Fist freeze is hold-to-freeze.
        action = next((g for g in ("pinch", "palm", "fist") if g in observations), "neutral")
        if action == self.gesture_candidate:
            self.gesture_count += 1
        else:
            self.gesture_candidate, self.gesture_count = action, 1
        if self.gesture_count >= 2:
            self.gesture_frozen = action == "fist"
            if action != self.gesture_latched:
                self.gesture_latched = action
                if now - self.gesture_action_time > 0.8:
                    if action == "pinch":
                        self.mode_index = (self.mode_index + 1) % len(self.MODES)
                        self.notify(f"Pinch: {self.mode}")
                        self.gesture_action_time = now
                    elif action == "palm":
                        self.reset_canvas()
                        self.notify("Open palm: canvas cleared")
                        self.gesture_action_time = now
        if not hands:
            self.gesture_frozen = False
        self.hand_result_time = now

    def update_gestures(self, frame: np.ndarray, now: float) -> None:
        if not self.gestures_enabled:
            return
        try:
            if self.hand_future is not None and self.hand_future.done():
                self.consume_hands(self.hand_future.result(), now)
                self.hand_future = None
            if now - self.hand_result_time > 0.6:
                self.gesture_frozen = self.two_hands = False
                self.finger_points.clear()
            if self.hand_future is None and now - self.hand_submit_time >= 1 / 12:
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(cv2.resize(frame, (320, max(1, round(h * 320 / w)))),
                                   cv2.COLOR_BGR2RGB)
                self.hand_timestamp = max(self.hand_timestamp + 1, int(now * 1000))
                self.hand_future = self.hand_executor.submit(
                    detect_hands_local, self.hand_model_path, rgb, self.hand_timestamp)
                self.hand_submit_time = now
        except Exception as error:
            self.close_gestures()
            self.notify(f"Gesture tracking stopped: {error}")

    def audio_callback(self, data, frames, callback_time, status) -> None:
        # Keep real-time callback work minimal: no FFT, OpenCV, files, or printing.
        if status:
            self.audio_status = str(status)
        self.audio_blocks.append((time.perf_counter(), data[:, 0].copy()))

    def toggle_audio(self) -> None:
        if self.audio_enabled:
            self.close_audio()
            self.notify("Audio OFF")
            return
        try:
            import sounddevice as sd
            device = sd.query_devices(kind="input")
            self.audio_rate = float(device["default_samplerate"])
            self.audio_stream = sd.InputStream(samplerate=self.audio_rate, channels=1,
                                               dtype="float32", blocksize=2048,
                                               callback=self.audio_callback)
            self.audio_stream.start()
            self.audio_enabled = True
            self.notify("Audio ON: local default microphone")
        except Exception as error:
            self.close_audio()
            self.notify(f"Audio unavailable: {error}")

    def close_audio(self) -> None:
        if self.audio_stream is not None:
            try:
                self.audio_stream.stop()
            except Exception as error:
                print(f"Audio stop: {error}")
            finally:
                try:
                    self.audio_stream.close()
                except Exception as error:
                    print(f"Audio close: {error}")
                self.audio_stream = None
        self.audio_enabled = False
        self.audio_blocks.clear()
        self.bass = self.treble = 0.0
        self.bass_average = 0.001
        self.rings.clear()

    def analyze_audio(self, samples: np.ndarray, now: float) -> bool:
        if samples.size < 64:
            return False
        samples = np.nan_to_num(samples.astype(np.float32))
        samples -= samples.mean()
        spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) * 2 / samples.size
        frequencies = np.fft.rfftfreq(samples.size, 1 / self.audio_rate)
        low = spectrum[(frequencies >= 40) & (frequencies <= 250)]
        high = spectrum[(frequencies >= 2000) & (frequencies <= 10000)]
        energy = float(np.linalg.norm(low)) if low.size else 0.0
        treble = float(np.linalg.norm(high)) if high.size else 0.0
        beat = (energy > max(0.008, self.bass_average * 1.65)
                and now - self.beat_time > 0.3)
        self.bass_average = self.bass_average * 0.92 + energy * 0.08
        self.bass = self.bass * 0.65 + min(1.0, energy * 10) * 0.35
        self.treble = self.treble * 0.65 + min(1.0, treble * 16) * 0.35
        if beat:
            self.beat_time = now
        return beat

    def update_audio(self, now: float, emit: bool = True) -> None:
        if not self.audio_enabled:
            return
        if self.audio_stream is not None and not self.audio_stream.active:
            self.close_audio()
            self.notify("Audio stream stopped; press A to retry")
            return
        if self.audio_status:
            self.notify(f"Audio: {self.audio_status.strip()}")
            self.audio_status = ""
        beat = False
        for _ in range(4):
            try:
                timestamp, block = self.audio_blocks.popleft()
            except IndexError:
                break
            if now - timestamp < 0.5:
                beat = self.analyze_audio(block, timestamp) or beat
        self.bass *= 0.98
        self.treble *= 0.98
        if beat and emit:
            self.rings.append([0.0, 1.0])
            self.rings = self.rings[-8:]

    def particle_layer(self, mask: np.ndarray, elapsed: float, now: float) -> np.ndarray:
        layer = np.zeros_like(self.canvas)
        dt = min(0.1, max(0.001, elapsed))
        h, w = mask.shape
        self.spawn_credit = min(12, self.spawn_credit + dt * 100)
        if len(self.particles) < 450 and self.spawn_credit >= 1:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted((c for c in contours if cv2.contourArea(c) >= 12),
                              key=cv2.contourArea, reverse=True)[:12]
            if contours:
                points = np.concatenate([c.reshape(-1, 2) for c in contours])
                count = min(int(self.spawn_credit), 450 - len(self.particles))
                new = np.empty((count, 9), np.float32)
                new[:, :2] = points[self.rng.integers(0, len(points), count)]
                new[:, 2:4] = self.rng.normal(0, 38, (count, 2))
                new[:, 3] -= 25
                new[:, 4] = self.rng.uniform(0.5, 1.7, count)
                new[:, 5] = new[:, 4]
                new[:, 6:9] = self.hue_color(now * 35)
                self.particles = np.concatenate((self.particles, new))
                self.spawn_credit -= count
        p = self.particles
        if p.size:
            p[:, 3] += 45 * dt  # gravity in working pixels / second squared
            p[:, :2] += p[:, 2:4] * dt
            p[:, 4] -= dt
            p = p[(p[:, 4] > 0) & (p[:, 0] >= 0) & (p[:, 0] < w)
                  & (p[:, 1] >= 0) & (p[:, 1] < h)]
            self.particles = p
            xy = p[:, :2].astype(np.int32)
            colors = p[:, 6:9] * (p[:, 4:5] / p[:, 5:6])
            np.add.at(layer, (xy[:, 1], xy[:, 0]), colors)
            layer = cv2.GaussianBlur(layer, (0, 0), 0.8) * 4
        return layer

    def audio_layer(self, edges: np.ndarray, elapsed: float) -> np.ndarray:
        layer = np.zeros_like(self.canvas)
        h, w = edges.shape
        dt = min(0.1, max(0.001, elapsed))
        alive = []
        for ring in self.rings:
            ring[0] += dt * max(w, h) * 0.65
            ring[1] -= dt * 0.85
            if ring[1] > 0:
                cv2.circle(layer, (w // 2, h // 2), int(ring[0]),
                           (255 * ring[1], 160 * ring[1], 230 * ring[1]), 2, cv2.LINE_AA)
                alive.append(ring)
        self.rings = alive
        if self.treble > 0.02:
            ys, xs = np.nonzero(edges)
            if xs.size:
                count = min(60, int(self.treble * 60) + 1)
                indices = self.rng.integers(0, xs.size, count)
                np.add.at(layer, (ys[indices], xs[indices]),
                          np.array([255, 230, 255], np.float32) * self.treble)
        return layer

    def apply_kaleidoscope(self, image: np.ndarray) -> np.ndarray:
        ways = self.kaleidoscope or (6 if self.two_hands else 0)
        if not ways:
            return image
        if ways not in self.kaleidoscope_maps:
            h, w = image.shape[:2]
            y, x = np.indices((h, w), dtype=np.float32)
            x -= (w - 1) / 2
            y -= (h - 1) / 2
            radius = np.hypot(x, y)
            wedge = 2 * np.pi / ways
            angle = np.mod(np.arctan2(y, x), wedge)
            angle = np.minimum(angle, wedge - angle)
            map_x = (radius * np.cos(angle) + (w - 1) / 2).astype(np.float32)
            map_y = (radius * np.sin(angle) + (h - 1) / 2).astype(np.float32)
            self.kaleidoscope_maps[ways] = (map_x, map_y)
        return cv2.remap(image, *self.kaleidoscope_maps[ways], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    def apply_time_slice(self, image: np.ndarray, elapsed: float) -> np.ndarray:
        if not self.time_slice:
            return image
        if self.slice_canvas is None:
            self.slice_canvas = image.copy()
            self.slice_position = 0.0
        w = image.shape[1]
        advance = min(w, max(1, round(w * min(elapsed, 0.1) / 4)))
        start = int(self.slice_position) % w
        columns = (np.arange(advance) + start) % w
        self.slice_canvas[:, columns] = image[:, columns]
        self.slice_position = (start + advance) % w
        return self.slice_canvas.copy()

    def apply_preset(self, number: int) -> None:
        # Audio / gesture device permission stays under explicit A / G control.
        settings = ((0, .96, 1.4, False, 0, False),
                    (1, .975, 1.8, True, 0, False),
                    (4, .97, 1.6, True, 6, False),
                    (2, .985, 1.0, False, 0, True),
                    (3, .95, 2.0, True, 4, False))
        (self.mode_index, self.decay, self.glow, self.particles_enabled,
         self.kaleidoscope, self.time_slice) = settings[number - 1]
        self.reset_canvas()
        self.notify(f"Preset {number}: {self.mode}")

    def blend_trails(self, trace: np.ndarray, elapsed: float) -> np.ndarray:
        self.canvas *= self.decay ** (max(0.001, elapsed) * 30)
        # Lighten preserves previous colors without rapid additive saturation.
        np.maximum(self.canvas, trace, out=self.canvas)
        return self.canvas

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        now = time.perf_counter()
        elapsed = 1 / 30 if self.last_process_time is None else now - self.last_process_time
        self.last_process_time = now
        if self.mirror:
            frame = cv2.flip(frame, 1)
        work_size = self.prepare_buffers(frame)
        small = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
        self.update_gestures(small, now)
        held = self.paused or self.frozen or self.gesture_frozen
        self.update_audio(now, emit=not held)
        self.effective_glow = min(4.0, self.glow + self.bass * 2.5)
        if held and self.effect_frame is not None:
            self.last_process_time = None
            self.previous_gray = None
            self.stroke_layer.fill(0)
            self.finger_points.clear()
            return self.effect_frame.copy()
        mask = self.apply_motion_mask(small)
        edges = cv2.Canny(self.previous_gray, 65, 140)
        trace = self.colorize(small, mask, edges, now) + self.stroke_layer
        self.stroke_layer.fill(0)
        trace = self.apply_glow(trace)
        trails = self.blend_trails(trace, elapsed)
        additive = np.zeros_like(trails)
        if self.particles_enabled:
            additive += self.particle_layer(mask, elapsed, now)
        if self.audio_enabled:
            additive += self.audio_layer(edges, elapsed)
        if self.particles_enabled or self.audio_enabled:
            trails += self.apply_glow(additive) * min(1.0, elapsed * 30) * 0.5
        np.clip(trails, 0, 255, out=trails)
        display_trails = self.apply_kaleidoscope(trails)
        base_weight = 0.70 if self.mode == "Ghost" else 0.52
        edge_weight = 0.22 if self.mode == "Neon" else 0.055
        edge_weight += self.treble * 0.45
        edge_layer = edges[..., None].astype(np.float32) * np.array([1.0, 0.9, 0.55], np.float32)
        composite = (small.astype(np.float32) * base_weight + display_trails * 0.88
                     + edge_layer * edge_weight) * self.vignette
        composite = np.clip(composite, 0, 255).astype(np.uint8)
        composite = self.apply_time_slice(composite, elapsed)
        # Keep the camera's actual output dimensions; never stretch unsupported 720p.
        self.effect_frame = cv2.resize(composite, (frame.shape[1], frame.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)
        return self.effect_frame.copy()

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        if not self.show_hud:
            return output
        height, width = output.shape[:2]
        scale = min(1.0, width / 960, height / 540)
        # Render into a fixed-size transparent-style panel, then scale for small cameras.
        panel_width = 690
        lines = ["ECHOTRACE LIVE  /  LOCAL LIGHT PAINTING",
                 f"{self.fps:4.1f} FPS   {self.mode.upper()}   Decay {self.decay:.3f}   Glow {self.effective_glow:.1f}",
                 f"Mirror {'ON' if self.mirror else 'OFF'}   "
                 f"{'FROZEN' if self.frozen or self.gesture_frozen else ('PAUSED' if self.paused else 'LIVE')}   "
                 f"{'REC ' + str(int(time.perf_counter() - self.record_started)) + 's' if self.writer else 'H: help'}"]
        ways = self.kaleidoscope or (6 if self.two_hands else 0)
        lines += [f"G: {'ON' if self.gestures_enabled else 'OFF'}   A: {'ON' if self.audio_enabled else 'OFF'}   "
                  f"P: {'ON' if self.particles_enabled else 'OFF'}   K: {ways or 'OFF'}   "
                  f"T: {'ON' if self.time_slice else 'OFF'}"]
        if self.show_help:
            lines += ["Q / Esc: quit    S: snapshot    V: record MP4 / stop",
                      "C: mode    R: clear trails    M: mirror    Space: pause",
                      "+ / -: trail retention    [ / ]: glow    H: hide help",
                      "G: gestures    A: audio    P: particles    K: symmetry",
                      "T: slit-scan    F: freeze    1-5: presets    Z: hide HUD"]
        if time.perf_counter() < self.notice_until:
            # Keep long filenames out of the panel; full details remain in console.
            lines += [self.notice[:78]]
        panel = np.full((16 + len(lines) * 26, panel_width, 3), (22, 13, 13), np.uint8)
        for index, line in enumerate(lines):
            color = (240, 240, 240) if index else (255, 235, 110)
            cv2.putText(panel, line, (12, 25 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, color, 1, cv2.LINE_AA)
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ph, pw = panel.shape[:2]
        margin = max(2, int(12 * scale))
        ph = min(ph, height - margin)
        pw = min(pw, width - margin)
        region = output[margin:margin + ph, margin:margin + pw]
        cv2.addWeighted(region, 0.22, panel[:ph, :pw], 0.78, 0, dst=region)
        return output

    def save_snapshot(self, frame: np.ndarray | None = None) -> None:
        frame = self.latest_result if frame is None else frame
        if frame is None:
            self.notify("No frame available to save yet.")
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"echotrace_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
            if not cv2.imwrite(str(path), frame):
                raise OSError("OpenCV could not write the image")
            self.notify(f"Snapshot saved: {path.name}")
        except (OSError, cv2.error) as error:
            self.notify(f"Snapshot failed: {error}")

    def stop_recording(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.notify(f"Recording saved: {self.record_path.name}")
        self.record_size = None

    def toggle_recording(self) -> None:
        if self.writer is not None:
            self.stop_recording()
            return
        if self.latest_result is None:
            self.notify("Wait for the first frame before recording.")
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            h, w = self.latest_result.shape[:2]
            # Common video codecs need even dimensions.
            size = (w + w % 2, h + h % 2)
            self.record_fps = min(30.0, max(5.0, self.fps))
            for codec, extension in (("mp4v", "mp4"), ("XVID", "avi"), ("MJPG", "avi")):
                path = self.output_dir / f"echotrace_{stamp}_{codec}.{extension}"
                writer = None
                try:
                    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec),
                                             self.record_fps, size)
                    if writer.isOpened():
                        self.writer = writer
                        self.record_path = path
                        self.record_size = size
                        self.record_started = time.perf_counter()
                        self.record_frames = 0
                        self.notify(f"Recording {extension.upper()} ({codec}): {path.name}")
                        return
                except cv2.error:
                    pass
                finally:
                    if writer is not None and writer is not self.writer:
                        writer.release()
                if path.exists():
                    path.unlink()
            self.notify("Recording failed: MP4, XVID and MJPG codecs unavailable.")
        except (OSError, cv2.error) as error:
            self.notify(f"Recording failed: {error}")

    def record_frame(self, frame: np.ndarray) -> None:
        if self.writer is None:
            return
        try:
            # Constant-rate output paced by wall time. Pauses record a frozen frame.
            target = int((time.perf_counter() - self.record_started) * self.record_fps) + 1
            count = min(5, max(0, target - self.record_frames))
            if count:
                w, h = self.record_size
                video_frame = cv2.copyMakeBorder(frame, 0, h - frame.shape[0],
                                                 0, w - frame.shape[1], cv2.BORDER_REPLICATE)
                for _ in range(count):
                    self.writer.write(video_frame)
                self.record_frames += count
        except cv2.error as error:
            self.stop_recording()
            self.notify(f"Recording interrupted: {error}")

    def handle_key(self, key: int) -> None:
        if key < 0:
            return
        key &= 0xFF
        if key in (27, ord("q"), ord("Q")):
            self.running = False
        elif key in (ord("s"), ord("S")):
            self.save_snapshot()
        elif key in (ord("v"), ord("V")):
            self.toggle_recording()
        elif key in (ord("r"), ord("R")):
            self.reset_canvas()
            self.notify("Trail canvas reset")
        elif key in (ord("c"), ord("C")):
            self.mode_index = (self.mode_index + 1) % len(self.MODES)
            self.notify(f"Mode: {self.mode}")
        elif key in (ord("+"), ord("=")):
            self.decay = min(0.995, round(self.decay + 0.005, 3))
        elif key in (ord("-"), ord("_")):
            self.decay = max(0.80, round(self.decay - 0.005, 3))
        elif key == ord("["):
            self.glow = max(0.0, round(self.glow - 0.2, 1))
        elif key == ord("]"):
            self.glow = min(4.0, round(self.glow + 0.2, 1))
        elif key in (ord("m"), ord("M")):
            self.mirror = not self.mirror
            self.reset_canvas()
        elif key == ord(" "):
            self.paused = not self.paused
            self.previous_gray = None
            self.last_process_time = None
        elif key in (ord("h"), ord("H")):
            self.show_help = not self.show_help
        elif key in (ord("g"), ord("G")):
            self.toggle_gestures()
        elif key in (ord("a"), ord("A")):
            self.toggle_audio()
        elif key in (ord("p"), ord("P")):
            self.particles_enabled = not self.particles_enabled
            self.particles = np.empty((0, 9), np.float32)
        elif key in (ord("k"), ord("K")):
            options = (0, 2, 4, 6, 8)
            self.kaleidoscope = options[(options.index(self.kaleidoscope) + 1) % len(options)]
        elif key in (ord("t"), ord("T")):
            self.time_slice = not self.time_slice
            self.slice_canvas = None
        elif key in (ord("f"), ord("F")):
            self.frozen = not self.frozen
            self.previous_gray = None
            self.last_process_time = None
        elif key in (ord("z"), ord("Z")):
            self.show_hud = not self.show_hud
        elif ord("1") <= key <= ord("5"):
            self.apply_preset(key - ord("0"))

    def run(self) -> int:
        status = 0
        try:
            if not self.open_camera():
                return 1
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            failures = 0
            last_tick = time.perf_counter()
            print("EchoTrace Live — focus the window for controls. H: help. Q: quit.")
            while self.running:
                start = time.perf_counter()
                # Continue reading while paused so resuming uses fresh camera frames.
                ok, frame = self.capture.read()
                if not ok or frame is None or frame.size == 0:
                    failures += 1
                    if failures >= 30:
                        print("ERROR: Webcam stopped returning frames. Check permissions, "
                              "connection, and other apps using the camera.")
                        status = 1
                        break
                    self.handle_key(cv2.waitKey(30))
                    if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue
                failures = 0
                result = self.process_frame(frame)
                self.latest_result = self.draw_hud(result)
                self.record_frame(self.latest_result)
                cv2.imshow(self.WINDOW, self.latest_result)
                # Limit display loop to roughly 30 Hz without delaying slow cameras.
                delay = max(1, math.ceil((1 / 30 - (time.perf_counter() - start)) * 1000))
                self.handle_key(cv2.waitKey(delay))
                if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                now = time.perf_counter()
                self.fps = self.fps * 0.9 + (1 / max(now - last_tick, 0.001)) * 0.1
                last_tick = now
        except KeyboardInterrupt:
            pass
        except cv2.error as error:
            print(f"OpenCV error: {error}\nCheck camera permissions and install the desktop "
                  "opencv-contrib-python desktop package (not a headless wheel).")
            status = 1
        finally:
            self.close_audio()
            self.close_gestures()
            self.stop_recording()
            if self.capture is not None:
                self.capture.release()
            cv2.destroyAllWindows()
        return status


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(EchoTraceAgent().run())
