"""MediaPipe hand gestures, fingertip drawing, and the hands-only paint mask."""

from __future__ import annotations

import importlib.util
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import cv2
import numpy as np

from .paths import MODEL_PATH

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


class GestureMixin:
    # Landmarks arrive at ~12 Hz and lag fast hands, so masks/gestures expire after this.
    HAND_STALE_SECONDS = 0.6

    def toggle_gestures(self) -> None:
        if self.gestures_enabled:
            self.close_gestures()
            self.notify("Gestures OFF")
            return
        try:
            if importlib.util.find_spec("mediapipe") is None:
                raise ImportError("Install mediapipe from requirements.txt")
            if not MODEL_PATH.is_file():
                raise FileNotFoundError("Missing models/hand_landmarker.task; see README")
            self.hand_model_path = str(MODEL_PATH)
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
        # Hands-only painting needs the tracker, so it switches off with it.
        self.hands_only = False
        self.hand_mask = None

    def toggle_hands_only(self) -> None:
        if self.hands_only:
            self.hands_only = False
            self.notify("Hands-only OFF: all motion paints")
            return
        if not self.gestures_enabled:
            self.toggle_gestures()
            if not self.gestures_enabled:
                return
        self.hands_only = True
        self.reset_canvas()
        self.notify("Hands-only ON: only tracked hands paint")

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

    def build_hand_mask(self, hands: list[np.ndarray], width: int, height: int) -> np.ndarray:
        mask = np.zeros((height, width), np.uint8)
        for points in hands:
            pixels = np.round(points[:, :2] * [width - 1, height - 1]).astype(np.int32)
            cv2.fillConvexPoly(mask, cv2.convexHull(pixels), 255)
        if hands:
            # Generous margin: the hand keeps moving between ~12 Hz detections.
            size = max(9, int(width * 0.08)) | 1
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
        return mask

    def restrict_to_hands(self, mask: np.ndarray, now: float) -> np.ndarray:
        if not self.hands_only:
            return mask
        if (self.hand_mask is None or self.hand_mask.shape != mask.shape
                or now - self.hand_mask_time > self.HAND_STALE_SECONDS):
            return np.zeros_like(mask)
        return cv2.bitwise_and(mask, self.hand_mask)

    def consume_hands(self, result, now: float) -> None:
        hands = result.hand_landmarks
        self.two_hands = len(hands) >= 2
        observations = []
        hand_points = []
        next_points = {}
        height, width = self.canvas.shape[:2]
        for index, landmarks in enumerate(hands):
            gesture, points = self.classify_hand(landmarks)
            observations.append(gesture)
            hand_points.append(points)
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
        self.hand_mask = self.build_hand_mask(hand_points, width, height) if hands else None
        self.hand_mask_time = now
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
            if now - self.hand_result_time > self.HAND_STALE_SECONDS:
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
