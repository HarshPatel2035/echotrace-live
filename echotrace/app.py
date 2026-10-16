"""EchoTrace Live: entirely local, CPU-friendly webcam light painting."""

from __future__ import annotations

import platform
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .audio import AudioMixin
from .camera import FrameSource
from .capture import CaptureMixin
from .effects import EffectsMixin
from .gestures import GestureMixin
from .hud import HudMixin
from .paths import CAPTURES_DIR


class EchoTraceAgent(EffectsMixin, GestureMixin, AudioMixin, CaptureMixin, HudMixin):
    WINDOW = "EchoTrace Live"
    MODES = ("Echo", "Neon", "Ghost", "Fire", "Rainbow")
    PRESETS = ((0, .96, 1.4, False, 0, False),
               (1, .975, 1.8, True, 0, False),
               (4, .97, 1.6, True, 6, False),
               (2, .985, 1.0, False, 0, True),
               (3, .95, 2.0, True, 4, False))

    def __init__(self, camera: int = 0, width: int = 1280, height: int = 720,
                 fps: int = 30, work_width: int = 640, mirror: bool = True,
                 capture_hud: bool = False, flicker_limit: float = 0.45,
                 output_dir: Path | None = None) -> None:
        self.camera_index = camera
        self.request_size = (width, height)
        self.request_fps = fps
        self.work_width = work_width
        self.capture = None
        self.writer = None
        self.record_path = None
        self.record_size = None
        self.record_started = 0.0
        self.record_frames = 0
        self.record_fps = 30.0
        self.output_dir = output_dir or CAPTURES_DIR
        self.mode_index = 0
        # Retention per 1/30 second: larger numbers mean longer trails.
        self.decay = 0.96
        self.glow = 1.4
        self.mirror = mirror
        self.paused = False
        self.show_help = False
        self.running = True
        # Threaded main loop: keys go from the window thread to the effects thread,
        # finished frames come back under output_lock.
        self.pending_keys = deque()
        self.output_lock = threading.Lock()
        self.output_sequence = 0
        self.worker_error = None
        self.camera_failed = False
        self.canvas = None
        self.previous_gray = None
        self.motion_speed = None
        # Fraction of the frame that may "move" at once before it counts as flicker.
        self.flicker_limit = flicker_limit
        self.flicker_notice_time = 0.0
        self.vignette = None
        self.frame_shape = None
        self.latest_result = None
        self.latest_clean = None
        self.capture_hud = capture_hud
        self.fps = 30.0
        self.last_process_time = None
        self.notice = "Move your hands to paint with light  -  H: help"
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
        self.hand_mask = None
        self.hand_mask_time = 0.0
        self.hands_only = False
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
        self.capture = cv2.VideoCapture(self.camera_index)
        if not self.capture.isOpened() and platform.system() == "Windows":
            self.capture.release()
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            print(f"ERROR: Could not open webcam {self.camera_index}. Check camera permissions "
                  "and close other camera apps. See README.md for troubleshooting.")
            return False
        # These are requests; use the actual frame size returned by the camera.
        # A width/height/fps of 0 keeps the camera's own default.
        width, height = self.request_size
        if width and height:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if self.request_fps:
            self.capture.set(cv2.CAP_PROP_FPS, self.request_fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def reset_canvas(self) -> None:
        if self.canvas is not None:
            self.canvas.fill(0)
        self.previous_gray = None
        self.motion_speed = None
        self.last_process_time = None
        self.particles = np.empty((0, 9), np.float32)
        self.rings.clear()
        self.finger_points.clear()
        if self.stroke_layer is not None:
            self.stroke_layer.fill(0)
        self.slice_canvas = None
        self.slice_position = 0.0

    def apply_preset(self, number: int) -> None:
        # Audio / gesture device permission stays under explicit A / G control.
        (self.mode_index, self.decay, self.glow, self.particles_enabled,
         self.kaleidoscope, self.time_slice) = self.PRESETS[number - 1]
        self.reset_canvas()
        self.notify(f"Preset {number}: {self.mode}")

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
        mask = self.restrict_to_hands(self.apply_motion_mask(small), now)
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
        overlay = np.clip(display_trails * 0.88 + edge_layer * edge_weight, 0, 255).astype(np.uint8)
        # Only the light overlay is upscaled; the camera image stays at full resolution.
        height, width = frame.shape[:2]
        if overlay.shape[:2] != (height, width):
            overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_LINEAR)
        composite = cv2.addWeighted(frame, base_weight, overlay, 1.0, 0)
        composite = cv2.multiply(composite, self.vignette, scale=1 / 255)
        self.effect_frame = self.apply_time_slice(composite, elapsed)
        return self.effect_frame.copy()

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
        elif key in (ord("o"), ord("O")):
            self.toggle_hands_only()
        elif key in (ord("i"), ord("I")):
            self.capture_hud = not self.capture_hud
            self.notify(f"HUD in snapshots/recordings: {'ON' if self.capture_hud else 'OFF'}")
        elif ord("1") <= key <= ord("5"):
            self.apply_preset(key - ord("0"))

    def processing_loop(self, source: FrameSource) -> None:
        """Effects, HUD, and recording, run beside the display so their costs overlap."""
        try:
            sequence = 0
            last_tick = time.perf_counter()
            while self.running:
                # Keys are applied here, between frames, so they never race the pipeline.
                while self.pending_keys:
                    self.handle_key(self.pending_keys.popleft())
                newest, frame, failures = source.next_frame(sequence)
                if failures >= 30:
                    self.camera_failed = True
                    self.running = False
                    break
                if newest == sequence or frame is None:
                    continue
                sequence = newest
                clean = self.process_frame(frame)
                result = self.draw_hud(clean)
                with self.output_lock:
                    self.latest_clean, self.latest_result = clean, result
                    self.output_sequence += 1
                self.record_frame(self.capture_frame())
                now = time.perf_counter()
                self.fps = self.fps * 0.9 + (1 / max(now - last_tick, 0.001)) * 0.1
                last_tick = now
        except BaseException as error:
            self.worker_error = error
            self.running = False

    def run(self) -> int:
        status = 0
        source = worker = None
        try:
            if not self.open_camera():
                return 1
            cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
            print("EchoTrace Live — focus the window for controls. H: help. Q: quit.")
            # Camera reads, effects, and display each take ~15 ms. Run in sequence they
            # exceed the 33 ms frame budget, so each gets its own thread. The window
            # stays on the main thread, which macOS requires.
            source = FrameSource(self.capture)
            source.start()
            worker = threading.Thread(target=self.processing_loop, args=(source,),
                                      name="EchoTraceEffects", daemon=True)
            worker.start()
            shown = 0
            while self.running:
                with self.output_lock:
                    sequence, image = self.output_sequence, self.latest_result
                if sequence != shown and image is not None:
                    cv2.imshow(self.WINDOW, image)
                    shown = sequence
                    key = cv2.waitKey(1)
                else:
                    key = cv2.waitKey(5)
                if key >= 0:
                    self.pending_keys.append(key)
                if cv2.getWindowProperty(self.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if self.worker_error is not None:
                raise self.worker_error
            if self.camera_failed:
                print("ERROR: Webcam stopped returning frames. Check permissions, "
                      "connection, and other apps using the camera.")
                status = 1
        except KeyboardInterrupt:
            pass
        except cv2.error as error:
            print(f"OpenCV error: {error}\nCheck camera permissions and install the desktop "
                  "opencv-contrib-python desktop package (not a headless wheel).")
            status = 1
        finally:
            self.running = False
            if worker is not None:
                worker.join(timeout=2.0)
            if source is not None:
                source.stop()
            self.close_audio()
            self.close_gestures()
            self.stop_recording()
            if self.capture is not None:
                self.capture.release()
            cv2.destroyAllWindows()
        return status
