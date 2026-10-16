"""Pixel effects: motion detection, colour modes, glow, particles, symmetry, slit-scan."""

from __future__ import annotations

import time

import cv2
import numpy as np


def hue_color(hue: float) -> np.ndarray:
    pixel = np.array([[[int(hue) % 180, 235, 255]]], np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


class EffectsMixin:
    hue_color = staticmethod(hue_color)

    def prepare_buffers(self, frame: np.ndarray) -> tuple[int, int]:
        height, width = frame.shape[:2]
        # Motion, persistent float canvas, and blur run at <= work_width pixels wide.
        scale = min(1.0, self.work_width / width)
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
            self.hand_mask = None
            # The vignette is applied to the full-resolution composite as a uint8 gain.
            y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
            x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
            gain = np.clip(1.0 - 0.18 * (x * x + y * y), 0.65, 1) * 255
            self.vignette = cv2.merge([np.round(gain).astype(np.uint8)] * 3)
        return work_width, work_height

    def apply_motion_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.previous_gray is None:
            mask = np.zeros(gray.shape, np.uint8)
            speed = np.zeros(gray.shape, np.float32)
        else:
            difference = cv2.absdiff(gray, self.previous_gray)
            _, mask = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            if self.flicker_limit and np.count_nonzero(mask) > mask.size * self.flicker_limit:
                # Most of the picture changed at once: auto-exposure, lights, or a bumped
                # camera. Painting it would flood the canvas, so skip this frame's motion.
                mask[:] = 0
                speed = np.zeros(gray.shape, np.float32)
                now = time.perf_counter()
                if now - self.flicker_notice_time > 5.0:
                    self.flicker_notice_time = now
                    self.notify("Whole-frame change ignored (exposure or camera movement)")
            else:
                mask = cv2.dilate(mask, None, iterations=1)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                # Unthresholded difference: a rough per-pixel motion speed in 0..1.
                smooth = cv2.GaussianBlur(difference, (9, 9), 0).astype(np.float32)
                speed = np.clip((smooth - 18) / 50, 0, 1)
        self.motion_speed = speed
        self.previous_gray = gray
        return mask

    def colorize(self, frame: np.ndarray, mask: np.ndarray,
                 edges: np.ndarray, now: float) -> np.ndarray:
        strength = mask.astype(np.float32)[..., None] / 255.0
        if self.mode == "Neon":
            # Slow motion glows cyan, fast motion glows magenta.
            speed = self.motion_speed
            if speed is None or speed.shape != mask.shape:
                speed = strength[..., 0]
            speed = speed[..., None]
            cyan = np.array([255, 255, 20], np.float32)
            magenta = np.array([255, 20, 255], np.float32)
            color = cyan * (1 - speed) + magenta * speed
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

    def blend_trails(self, trace: np.ndarray, elapsed: float) -> np.ndarray:
        self.canvas *= self.decay ** (max(0.001, elapsed) * 30)
        # Lighten preserves previous colors without rapid additive saturation.
        np.maximum(self.canvas, trace, out=self.canvas)
        return self.canvas

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
        if self.slice_canvas is None or self.slice_canvas.shape != image.shape:
            self.slice_canvas = image.copy()
            self.slice_position = 0.0
        w = image.shape[1]
        advance = min(w, max(1, round(w * min(elapsed, 0.1) / 4)))
        start = int(self.slice_position) % w
        columns = (np.arange(advance) + start) % w
        self.slice_canvas[:, columns] = image[:, columns]
        self.slice_position = (start + advance) % w
        return self.slice_canvas.copy()
