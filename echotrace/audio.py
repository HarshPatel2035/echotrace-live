"""Microphone analysis: bass boosts glow, beats emit rings, treble adds sparkles."""

from __future__ import annotations

import time

import cv2
import numpy as np


class AudioMixin:
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
