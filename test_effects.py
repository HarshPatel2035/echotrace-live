"""Hardware-free regression checks: python -m unittest -v test_effects.py."""
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from main import EchoTraceAgent


def hand(gesture):
    """Geometric fixtures with a palm, curled/extended fingers and thumb."""
    points = np.zeros((21, 3), np.float32)
    points[0, :2] = (.5, .9)
    points[9, :2] = (.5, .6)
    points[4, :2] = (.1, .6)
    for tip, pip, x in ((8, 6, .4), (12, 10, .5), (16, 14, .6), (20, 18, .7)):
        points[pip, :2] = (x, .6)
        extended = gesture == "palm" or (gesture in ("index", "pinch") and tip == 8)
        points[tip, :2] = (x, .25 if extended else .78)
    if gesture == "pinch":
        points[4] = points[8] + (.015, .015, 0)
    return [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points]


def result(*gestures):
    return SimpleNamespace(hand_landmarks=[hand(g) for g in gestures],
                           handedness=[[SimpleNamespace(category_name=name)]
                                       for name in ("Left", "Right")[:len(gestures)]])


class EffectsTests(unittest.TestCase):
    def setUp(self):
        self.agent = EchoTraceAgent()
        self.frame = np.zeros((360, 640, 3), np.uint8)
        self.agent.prepare_buffers(self.frame)
        self.agent.rng = np.random.default_rng(42)

    def tearDown(self):
        self.agent.close_audio()
        self.agent.close_gestures()
        self.agent.stop_recording()

    def test_five_modes_no_motion_and_motion(self):
        for index in range(5):
            self.agent.mode_index = index
            self.agent.reset_canvas()
            self.agent.process_frame(self.frame)
            moving = self.frame.copy()
            moving[80:150, 160:240] = 255
            out = self.agent.process_frame(moving)
            self.assertEqual(out.shape, self.frame.shape)
            self.assertEqual(out.dtype, np.uint8)
            self.assertGreater(self.agent.canvas.max(), 0)
            previous = self.agent.canvas.copy()
            self.agent.process_frame(moving)
            self.assertTrue(np.all(self.agent.canvas <= previous + .001))

    def test_gestures_debounce_hold_and_release(self):
        for name in ("pinch", "palm", "fist", "index"):
            self.assertEqual(self.agent.classify_hand(hand(name))[0], name)
        self.agent.consume_hands(result("pinch"), 10)
        self.assertEqual(self.agent.mode_index, 0)
        self.agent.consume_hands(result("pinch"), 10.1)
        self.assertEqual(self.agent.mode_index, 1)
        self.agent.consume_hands(result("pinch"), 12)
        self.assertEqual(self.agent.mode_index, 1)
        self.agent.canvas.fill(100)
        self.agent.consume_hands(result("palm"), 13)
        self.agent.consume_hands(result("palm"), 13.1)
        self.assertEqual(self.agent.canvas.max(), 0)
        self.agent.consume_hands(result("fist"), 14)
        self.agent.consume_hands(result("fist"), 14.1)
        self.assertTrue(self.agent.gesture_frozen)
        self.agent.consume_hands(result(), 15)
        self.assertFalse(self.agent.gesture_frozen)
        self.agent.consume_hands(result("index", "index"), 16)
        self.assertTrue(self.agent.two_hands)
        self.assertGreater(self.agent.stroke_layer.max(), 0)
        self.assertEqual(len(self.agent.finger_points), 2)

    def test_freeze_is_exact_and_resume_detects_no_false_motion(self):
        expected = self.agent.process_frame(self.frame)
        self.agent.handle_key(ord("F"))
        changed = np.full_like(self.frame, 255)
        actual = self.agent.process_frame(changed)
        np.testing.assert_array_equal(actual, expected)
        self.agent.handle_key(ord("F"))
        self.agent.process_frame(changed)
        self.assertEqual(self.agent.canvas.max(), 0)

    def test_audio_frequency_response_and_beats(self):
        samples = np.arange(2048) / self.agent.audio_rate
        bass = (.2 * np.sin(2 * np.pi * 100 * samples)).astype(np.float32)
        self.assertTrue(self.agent.analyze_audio(bass, 10))
        self.assertGreater(self.agent.bass, .1)
        self.assertLess(self.agent.treble, .01)
        self.assertFalse(self.agent.analyze_audio(bass, 10.1))
        self.agent.bass = self.agent.treble = 0
        treble = (.2 * np.sin(2 * np.pi * 4000 * samples)).astype(np.float32)
        self.assertFalse(self.agent.analyze_audio(treble, 11))
        self.assertGreater(self.agent.treble, .1)
        self.assertLess(self.agent.bass, .01)
        self.agent.audio_enabled = True
        self.agent.bass_average = .001
        self.agent.audio_blocks.append((20, bass))
        self.agent.update_audio(20)
        self.assertEqual(len(self.agent.rings), 1)
        self.agent.audio_layer(np.zeros(self.frame.shape[:2], np.uint8), .1)
        self.assertGreater(self.agent.rings[0][0], 0)
        self.agent.audio_blocks.append((21, bass))
        self.agent.update_audio(21, emit=False)
        self.assertEqual(len(self.agent.rings), 1)
        for _ in range(100):
            self.agent.analyze_audio(np.zeros(2048, np.float32), 22)
        self.assertLess(self.agent.bass, .001)

    def test_particles_spawn_move_and_expire(self):
        mask = np.zeros(self.frame.shape[:2], np.uint8)
        mask[100:160, 120:180] = 255
        layer = self.agent.particle_layer(mask, .1, 10)
        self.assertGreater(layer.max(), 0)
        self.assertGreater(len(self.agent.particles), 0)
        before = self.agent.particles.copy()
        self.agent.particle_layer(np.zeros_like(mask), .1, 10.1)
        self.assertTrue(np.any(self.agent.particles[:, :2] != before[:, :2]))
        for _ in range(20):
            self.agent.particle_layer(np.zeros_like(mask), .1, 11)
        self.assertEqual(len(self.agent.particles), 0)
        for _ in range(150):
            self.agent.particle_layer(mask, .1, 12)
        self.assertLessEqual(len(self.agent.particles), 450)

    def test_kaleidoscope_preserves_canvas_and_caches_maps(self):
        square = self.agent.rng.uniform(0, 255, (100, 100, 3)).astype(np.float32)
        for ways in (2, 4, 6, 8):
            self.agent.kaleidoscope = ways
            output = self.agent.apply_kaleidoscope(square)
            self.assertEqual(output.shape, square.shape)
            np.testing.assert_allclose(output, np.rot90(output, 2), atol=9)
            maps = self.agent.kaleidoscope_maps[ways]
            self.agent.apply_kaleidoscope(square)
            self.assertIs(self.agent.kaleidoscope_maps[ways], maps)

    def test_slit_scan_changes_only_scanned_columns_and_wraps(self):
        self.agent.time_slice = True
        self.agent.apply_time_slice(self.frame, 1 / 30)
        white = np.full_like(self.frame, 255)
        out = self.agent.apply_time_slice(white, 1 / 30)
        self.assertGreater(np.count_nonzero(out), 0)
        self.assertLess(np.count_nonzero(out), out.size // 10)
        for _ in range(150):
            out = self.agent.apply_time_slice(white, 1 / 30)
        self.assertTrue(np.all(out == 255))
        self.agent.reset_canvas()
        self.assertIsNone(self.agent.slice_canvas)

    def test_controls_presets_and_hud(self):
        for key in "PKTZF":
            self.agent.handle_key(ord(key))
        self.assertTrue(self.agent.particles_enabled)
        self.assertEqual(self.agent.kaleidoscope, 2)
        self.assertTrue(self.agent.time_slice)
        self.assertTrue(self.agent.frozen)
        np.testing.assert_array_equal(self.agent.draw_hud(self.frame), self.frame)
        for number in range(1, 6):
            self.agent.handle_key(ord(str(number)))
            self.assertTrue(self.agent.frozen)
            self.assertEqual(self.agent.canvas.max(), 0)
        self.assertEqual(self.agent.mode, "Fire")
        self.assertEqual(self.agent.kaleidoscope, 4)

    def test_missing_optional_dependencies_are_nonfatal(self):
        with patch.dict("sys.modules", {"mediapipe": None, "sounddevice": None}):
            self.agent.toggle_gestures()
            self.agent.toggle_audio()
        self.assertFalse(self.agent.gestures_enabled)
        self.assertFalse(self.agent.audio_enabled)
        self.assertEqual(self.agent.process_frame(self.frame).shape, self.frame.shape)

    def test_hand_worker_failure_does_not_stop_effects(self):
        self.agent.gestures_enabled = True
        self.agent.gesture_frozen = True
        self.agent.hand_future = Future()
        self.agent.hand_future.set_exception(RuntimeError("native worker unavailable"))
        self.agent.update_gestures(self.frame, 10)
        self.assertFalse(self.agent.gestures_enabled)
        self.assertFalse(self.agent.gesture_frozen)
        self.assertEqual(self.agent.process_frame(self.frame).shape, self.frame.shape)

    def test_audio_device_lifecycle_and_open_failure(self):
        class Stream:
            active = False
            stopped = closed = False
            def start(self): self.active = True
            def stop(self): self.stopped = True
            def close(self): self.closed = True
        stream = Stream()
        device_api = SimpleNamespace(query_devices=lambda **kw: {"default_samplerate": 48000},
                                     InputStream=lambda **kw: stream)
        with patch.dict("sys.modules", {"sounddevice": device_api}):
            self.agent.toggle_audio()
            self.assertTrue(self.agent.audio_enabled)
            self.agent.audio_callback(np.ones((2048, 1), np.float32), 2048, None, None)
            self.assertEqual(len(self.agent.audio_blocks), 1)
            self.agent.toggle_audio()
            self.assertTrue(stream.stopped and stream.closed)
            self.assertFalse(self.agent.audio_enabled)
            device_api.query_devices = lambda **kw: (_ for _ in ()).throw(OSError("no microphone"))
            self.agent.toggle_audio()
            self.assertFalse(self.agent.audio_enabled)

    def test_snapshot_and_recording_decode(self):
        self.agent.latest_result = self.agent.process_frame(self.frame)
        with TemporaryDirectory() as directory:
            self.agent.output_dir = Path(directory)
            self.agent.save_snapshot()
            self.assertEqual(len(list(Path(directory).glob("*.png"))), 1)
            self.agent.toggle_recording()
            self.assertIsNotNone(self.agent.writer)
            self.agent.record_frame(self.agent.latest_result)
            path = self.agent.record_path
            self.agent.stop_recording()
            video = cv2.VideoCapture(str(path))
            ok, decoded = video.read()
            video.release()
            self.assertTrue(ok)
            self.assertEqual(decoded.shape, self.frame.shape)


if __name__ == "__main__":
    unittest.main()
