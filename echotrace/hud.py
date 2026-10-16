"""On-screen status panel and key help."""

from __future__ import annotations

import time

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.48


def on_off(value) -> str:
    return "ON" if value else "OFF"


class HudMixin:
    def hud_lines(self) -> list[str]:
        held = self.frozen or self.gesture_frozen
        state = "FROZEN" if held else ("PAUSED" if self.paused else "LIVE")
        record = (f"REC {int(time.perf_counter() - self.record_started)}s"
                  if self.writer else "H: help")
        ways = self.kaleidoscope or (6 if self.two_hands else 0)
        lines = ["ECHOTRACE LIVE  /  LOCAL LIGHT PAINTING",
                 f"{self.fps:4.1f} FPS   {self.mode.upper()}   Decay {self.decay:.3f}   "
                 f"Glow {self.effective_glow:.1f}",
                 f"Mirror {on_off(self.mirror)}   {state}   {record}",
                 f"G: {on_off(self.gestures_enabled)}   A: {on_off(self.audio_enabled)}   "
                 f"P: {on_off(self.particles_enabled)}   K: {ways or 'OFF'}   "
                 f"T: {on_off(self.time_slice)}   O: {on_off(self.hands_only)}"]
        if self.show_help:
            lines += ["Q / Esc: quit    S: snapshot    V: record MP4 / stop",
                      "C: mode    R: clear trails    M: mirror    Space: pause",
                      "+ / -: trail retention    [ / ]: glow    H: hide help",
                      "G: gestures    A: audio    P: particles    K: symmetry",
                      "T: slit-scan    F: freeze    1-5: presets    Z: hide HUD",
                      f"O: hands-only paint    I: HUD in captures ({on_off(self.capture_hud)})"]
        if time.perf_counter() < self.notice_until:
            # Keep long filenames out of the panel; full details remain in console.
            lines += [self.notice[:78]]
        return lines

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        output = frame.copy()
        if not self.show_hud:
            return output
        height, width = output.shape[:2]
        scale = min(1.0, width / 960, height / 540)
        lines = self.hud_lines()
        # Size the panel to its text so the compact HUD covers as little as possible.
        text_width = max(cv2.getTextSize(line, FONT, FONT_SCALE, 1)[0][0] for line in lines)
        panel = np.full((16 + len(lines) * 26, text_width + 24, 3), (22, 13, 13), np.uint8)
        for index, line in enumerate(lines):
            color = (240, 240, 240) if index else (255, 235, 110)
            cv2.putText(panel, line, (12, 25 + index * 26), FONT, FONT_SCALE, color, 1,
                        cv2.LINE_AA)
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ph, pw = panel.shape[:2]
        margin = max(2, int(12 * scale))
        ph = min(ph, height - margin)
        pw = min(pw, width - margin)
        region = output[margin:margin + ph, margin:margin + pw]
        cv2.addWeighted(region, 0.22, panel[:ph, :pw], 0.78, 0, dst=region)
        return output
