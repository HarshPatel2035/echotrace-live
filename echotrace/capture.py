"""PNG snapshots and paced video recording."""

from __future__ import annotations

import time
from datetime import datetime

import cv2
import numpy as np


class CaptureMixin:
    def capture_frame(self) -> np.ndarray | None:
        """The frame saved to disk: the clean effect image unless HUD capture is on."""
        if not self.capture_hud and self.latest_clean is not None:
            return self.latest_clean
        return self.latest_result

    def save_snapshot(self, frame: np.ndarray | None = None) -> None:
        frame = self.capture_frame() if frame is None else frame
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
