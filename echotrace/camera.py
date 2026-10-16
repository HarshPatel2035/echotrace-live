"""Background camera reader so the app never waits on capture.read()."""

from __future__ import annotations

import threading
import time

import numpy as np


class FrameSource(threading.Thread):
    """Reads frames continuously and keeps only the newest one."""

    def __init__(self, capture) -> None:
        super().__init__(name="EchoTraceCamera", daemon=True)
        self.capture = capture
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.sequence = 0
        self.failures = 0
        self.stopped = False

    def run(self) -> None:
        while not self.stopped:
            ok, frame = self.capture.read()
            good = ok and frame is not None and frame.size > 0
            with self.condition:
                if good:
                    self.frame = frame
                    self.sequence += 1
                    self.failures = 0
                else:
                    self.failures += 1
                self.condition.notify_all()
            if not good:
                time.sleep(0.03)

    def next_frame(self, last_sequence: int, timeout: float = 0.5):
        """Wait for a frame newer than last_sequence: returns (sequence, frame, failures)."""
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != last_sequence or self.stopped, timeout)
            return self.sequence, self.frame, self.failures

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.join(timeout=2.0)
