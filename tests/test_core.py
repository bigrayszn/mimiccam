"""Fast unit tests that need no camera, no GPU and no network.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mimiccam import config, credits, presets  # noqa: E402
from mimiccam.engine import PassthroughEngine  # noqa: E402


class OwnershipTests(unittest.TestCase):
    """Who the project belongs to, in one place, so it cannot drift."""

    def test_the_copyright_names_ray_bekson(self) -> None:
        import mimiccam

        self.assertIn("Ray Bekson", mimiccam.__copyright__)
        self.assertIn("bigrayszn", mimiccam.__author__)
        self.assertEqual(mimiccam.__url__, "https://github.com/bigrayszn/mimiccam")
        self.assertEqual(mimiccam.__license__, "GPL-2.0-or-later")

    def test_the_copyright_notice_names_ray_bekson(self) -> None:
        text = (Path(__file__).resolve().parents[1] / "COPYRIGHT.md").read_text(encoding="utf-8")
        self.assertIn("Ray Bekson", text)
        self.assertIn("bigrayszn/mimiccam", text)
        self.assertIn("GPL-2.0-or-later", text)

    def test_the_licence_file_is_the_verbatim_gpl(self) -> None:
        # No preamble: GitHub only declares "GPL-2.0" on the repository when the
        # file starts with the licence itself. The copyright lives in COPYRIGHT.md.
        text = (Path(__file__).resolve().parents[1] / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(text.lstrip().startswith("GNU GENERAL PUBLIC LICENSE"))
        self.assertNotIn("Ray Bekson", text)

    def test_version_flag_prints_the_owner(self) -> None:
        import contextlib
        import io

        from mimiccam.__main__ import main

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), self.assertRaises(SystemExit):
            main(["--version"])
        printed = buffer.getvalue()
        self.assertIn("Ray Bekson", printed)
        self.assertIn("bigrayszn/mimiccam", printed)


class PresetTests(unittest.TestCase):
    def test_character_preset_requires_reference(self) -> None:
        preset = presets.get("character")
        self.assertTrue(preset.needs_reference)
        self.assertIn("reference image", preset.prompt)

    def test_unknown_preset_falls_back(self) -> None:
        self.assertEqual(presets.get("nope").key, presets.DEFAULT_KEY)

    def test_custom_returns_user_text(self) -> None:
        self.assertEqual(presets.system_prompt("custom", "  make me a robot  "), "make me a robot")

    def test_extra_direction_is_appended(self) -> None:
        text = presets.system_prompt("anime", "add a wide-brim hat")
        self.assertIn("anime", text.lower())
        self.assertIn("add a wide-brim hat", text)

    def test_no_duplicate_when_prompt_is_unchanged(self) -> None:
        prompt = presets.get("suit").prompt
        self.assertEqual(presets.system_prompt("suit", prompt), prompt.strip())


class SettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._original = config.SETTINGS_PATH
        config.SETTINGS_PATH = Path(self._tmp.name) / "settings.json"

    def tearDown(self) -> None:
        config.SETTINGS_PATH = self._original
        self._tmp.cleanup()

    def test_roundtrip(self) -> None:
        settings = config.Settings(camera_index=1, mode="decart", prompt="hello")
        settings.save()
        loaded = config.Settings.load()
        self.assertEqual(loaded.camera_index, 1)
        self.assertEqual(loaded.mode, "decart")
        self.assertEqual(loaded.prompt, "hello")

    def test_missing_file_returns_defaults(self) -> None:
        self.assertEqual(config.Settings.load().fps, 30)

    def test_unknown_keys_are_ignored(self) -> None:
        config.SETTINGS_PATH.write_text('{"camera_index": 2, "bogus": 1}', encoding="utf-8")
        self.assertEqual(config.Settings.load().camera_index, 2)

    def test_picture_in_picture_defaults_on_and_persists_off(self) -> None:
        self.assertTrue(config.Settings.load().pip_camera)
        settings = config.Settings.load()
        settings.pip_camera = False
        settings.save()
        self.assertFalse(config.Settings.load().pip_camera)

    def test_key_is_masked(self) -> None:
        self.assertEqual(config.mask_key(""), "(none)")
        self.assertNotIn("secret", config.mask_key("ek_secret_value"))


class CreditMathTests(unittest.TestCase):
    """The balance is arithmetic, so it can be checked without a session."""

    def test_rate_follows_the_speed_setting(self) -> None:
        self.assertEqual(credits.usd_per_second(False), 0.02)
        self.assertEqual(credits.usd_per_second(True), 0.04)

    def test_spend_is_billed_seconds_times_rate(self) -> None:
        self.assertEqual(credits.usd_from_seconds(120, False), 2.4)
        self.assertEqual(credits.usd_from_seconds(60, True), 2.4)
        self.assertEqual(credits.usd_from_seconds(0, False), 0.0)

    def test_remaining_never_goes_negative(self) -> None:
        self.assertAlmostEqual(credits.remaining_usd(10.0, 2.23), 7.77)
        self.assertEqual(credits.remaining_usd(1.0, 4.0), 0.0)
        self.assertEqual(credits.remaining_usd(0.0, 0.0), 0.0)

    def test_money_and_clock_read_like_a_dashboard(self) -> None:
        self.assertEqual(credits.money(7.77), "$7.77")
        self.assertEqual(credits.money(0.004), "$0.004")
        self.assertEqual(credits.clock(0), "0:00")
        self.assertEqual(credits.clock(67), "1:07")

    def test_balance_and_spend_survive_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = config.SETTINGS_PATH
            config.SETTINGS_PATH = Path(tmp) / "settings.json"
            try:
                settings = config.Settings.load()
                self.assertEqual(settings.balance_usd, 0.0)
                settings.balance_usd = 10.0
                settings.balance_spent = 2.4
                settings.save()
                loaded = config.Settings.load()
                self.assertEqual(loaded.balance_usd, 10.0)
                self.assertEqual(loaded.balance_spent, 2.4)
            finally:
                config.SETTINGS_PATH = original


class PassthroughTests(unittest.TestCase):
    def test_latest_is_newest_frame(self) -> None:
        engine = PassthroughEngine()
        engine.start()
        first = np.zeros((4, 4, 3), dtype=np.uint8)
        second = np.full((4, 4, 3), 200, dtype=np.uint8)
        engine.push(first, 1.0)
        engine.push(second, 2.0)
        frame, ts = engine.latest()
        self.assertEqual(int(frame[0, 0, 0]), 200)
        engine.stop()

    def test_timestamp_is_a_fresh_perf_counter_reading(self) -> None:
        """The pipeline decides "model went quiet" by comparing this to perf_counter.

        Stamping the frame with the input timestamp would make a silent model look
        fresh forever, so the contract is: the timestamp is when the engine produced
        the frame, on the same clock the pipeline reads.
        """
        engine = PassthroughEngine()
        before = time.perf_counter()
        engine.push(np.zeros((4, 4, 3), dtype=np.uint8), 1.0)
        _, ts = engine.latest()
        after = time.perf_counter()
        self.assertGreaterEqual(ts, before)
        self.assertLessEqual(ts, after)

    def test_empty_engine_returns_none(self) -> None:
        frame, _ = PassthroughEngine().latest()
        self.assertIsNone(frame)


class NvidiaGateTests(unittest.TestCase):
    """"Strictly NVIDIA" is a product promise, so it is enforced, not just logged."""

    def test_missing_gpu_blocks_the_ai_engine(self) -> None:
        from mimiccam import gpu

        problem = gpu.require_nvidia(None)
        self.assertIsNotNone(problem)
        self.assertIn("NVIDIA", problem)
        self.assertIn("Passthrough", problem)

    def test_present_gpu_passes(self) -> None:
        from mimiccam import gpu

        info = {"name": "NVIDIA GeForce RTX 4070 Laptop GPU", "driver": "551.23",
                "vram_total_mb": 8188}
        self.assertIsNone(gpu.require_nvidia(info))
        self.assertIn("RTX 4070", gpu.describe(info))

    def test_missing_gpu_is_described_honestly(self) -> None:
        from mimiccam import gpu

        self.assertEqual(gpu.describe(None), "No NVIDIA GPU detected")


class AdapterPinTests(unittest.TestCase):
    """The models are pinned to the NVIDIA adapter, not to "whichever is 0"."""

    def test_adapters_are_enumerated_in_index_order(self) -> None:
        from mimiccam import gpu

        adapters = gpu.dxgi_adapters()
        if not adapters:
            self.skipTest("no DXGI adapters readable on this machine")
        self.assertEqual([a["index"] for a in adapters], list(range(len(adapters))))
        for adapter in adapters:
            self.assertTrue(adapter["name"])
            self.assertIsInstance(adapter["vendor"], int)

    def test_pinned_provider_names_the_nvidia_index(self) -> None:
        from mimiccam import gpu

        adapter = gpu.nvidia_adapter(refresh=True)
        if adapter is None:
            self.skipTest("no NVIDIA adapter on this machine")
        providers = gpu.directml_providers()
        self.assertEqual(providers[0][0], "DmlExecutionProvider")
        self.assertEqual(providers[0][1], {"device_id": adapter["index"]})
        self.assertEqual(providers[-1], "CPUExecutionProvider")
        self.assertEqual(gpu.adapter_name(providers), adapter["name"])

    def test_no_nvidia_adapter_is_a_refusal_not_a_fallback(self) -> None:
        from unittest import mock

        from mimiccam import gpu

        with mock.patch.object(gpu, "nvidia_adapter", return_value=None):
            with self.assertRaises(RuntimeError) as caught:
                gpu.directml_providers(strict=True)
            self.assertIn("NVIDIA", str(caught.exception))
            # Non-strict callers (tools, tests) still get a usable CPU list.
            self.assertEqual(gpu.directml_providers(), ["CPUExecutionProvider"])

    def test_adapter_name_falls_back_for_cpu_only(self) -> None:
        from mimiccam import gpu

        self.assertEqual(gpu.adapter_name(["CPUExecutionProvider"]), "CPU")
        self.assertEqual(gpu.adapter_name(None), "CPU")
        self.assertEqual(gpu.adapter_name([("DmlExecutionProvider", {})]), "CPU")

    def test_local_engine_refuses_without_the_nvidia_gpu(self) -> None:
        from unittest import mock

        from mimiccam import gpu
        from mimiccam.local_engine import LocalEngine, LocalEngineError

        engine = LocalEngine()  # no providers argument: strict is the default
        self.assertTrue(engine._strict_gpu)
        with mock.patch.object(
            gpu, "directml_providers", side_effect=RuntimeError("No NVIDIA DirectX adapter")
        ):
            with self.assertRaises(LocalEngineError) as caught:
                engine._setup()
        self.assertIn("NVIDIA", str(caught.exception))

    def test_explicit_providers_are_not_second_guessed(self) -> None:
        from mimiccam.local_engine import LocalEngine

        engine = LocalEngine(providers=["CPUExecutionProvider"])
        self.assertFalse(engine._strict_gpu)
        self.assertEqual(engine.providers, ["CPUExecutionProvider"])


class BillingPauseTests(unittest.TestCase):
    """Lucy is billed per connected second, so the engine must spend them well."""

    def _engine(self, **kwargs):
        from mimiccam.decart_engine import DecartEngine

        engine = DecartEngine(api_key="test-key", width=320, height=180, **kwargs)
        return engine

    @staticmethod
    def _tick(seconds: int):
        from types import SimpleNamespace

        return SimpleNamespace(seconds=seconds)

    def test_server_ticks_drive_the_meter(self) -> None:
        engine = self._engine()
        engine._on_generation_tick(self._tick(3))
        engine._on_generation_tick(self._tick(9))
        engine._on_generation_tick(self._tick(4))  # never runs backwards
        billing = engine.billing()
        self.assertEqual(billing["seconds"], 9.0)
        self.assertAlmostEqual(billing["usd"], 0.18, places=3)
        self.assertAlmostEqual(billing["usd_per_min"], 1.2, places=2)

    def test_fast_mode_costs_double(self) -> None:
        engine = self._engine(fast_mode=True)
        engine._on_generation_tick(self._tick(10))
        self.assertAlmostEqual(engine.billing()["usd"], 0.4, places=3)
        self.assertAlmostEqual(engine.billing()["usd_per_min"], 2.4, places=2)

    def test_parking_needs_idle_motion_and_a_settled_session(self) -> None:
        import time

        engine = self._engine(idle_after_s=30.0)
        now = time.perf_counter()
        engine._session_opened_at = now - 120.0

        engine._last_motion_ts = now  # someone just moved
        self.assertFalse(engine.should_park(now))

        engine._last_motion_ts = now - 31.0  # idle long enough
        self.assertTrue(engine.should_park(now))

        # A session that has only just started never parks: reconnecting would
        # bill its first seconds for nothing.
        engine._session_opened_at = now - 5.0
        self.assertFalse(engine.should_park(now))

    def test_parking_can_be_switched_off(self) -> None:
        import time

        engine = self._engine(pause_when_idle=False)
        now = time.perf_counter()
        engine._session_opened_at = now - 300.0
        engine._last_motion_ts = now - 299.0
        self.assertFalse(engine.should_park(now))

    def test_parked_session_shows_the_raw_camera(self) -> None:
        import numpy as np

        engine = self._engine()
        engine.parked = True
        frame = np.full((180, 320, 3), 90, dtype=np.uint8)
        engine.push(frame, 1.0)
        out, _ = engine.latest()
        self.assertIsNotNone(out)
        self.assertTrue(np.array_equal(out, frame))

    def test_supervisor_parks_then_reconnects(self) -> None:
        """The loop must pause, wait for movement, and start a fresh session."""
        import asyncio
        import time

        from mimiccam import decart_engine as de

        engine = self._engine()
        sessions: list[str] = []

        async def fake_session() -> None:
            sessions.append("session")
            if len(sessions) == 1:
                raise de._IdlePause()
            engine._stop.set()

        async def fake_wait() -> None:
            engine._last_motion_ts = time.perf_counter() + 1

        engine._session = fake_session
        engine._wait_for_motion = fake_wait
        engine._stop.clear()
        asyncio.run(engine._supervise())

        self.assertEqual(sessions, ["session", "session"])
        self.assertEqual(engine.pauses, 1)
        self.assertFalse(engine.parked)

    def test_motion_is_detected_and_stillness_is_not(self) -> None:
        import numpy as np

        engine = self._engine()
        still = np.full((180, 320, 3), 60, dtype=np.uint8)
        engine._last_thumb_at = 0.0
        engine.push(still, 1.0)
        settled = engine._last_motion_ts
        engine._last_thumb_at = 0.0
        engine.push(still.copy(), 1.1)
        self.assertEqual(engine._last_motion_ts, settled)

        moved = still.copy()
        moved[:, 160:] = 220  # half the frame changes
        engine._last_thumb_at = 0.0
        engine.push(moved, 1.2)
        self.assertGreater(engine._last_motion_ts, settled)


class CameraPreviewTests(unittest.TestCase):
    """Seeing your own camera is a separate job from seeing the model's output."""

    @staticmethod
    def _pipeline(seen, on_input=True):
        from mimiccam.pipeline import Pipeline

        frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in (10, 20, 30)]
        counter = {"i": 0, "pipeline": None}

        class FakeCapture:
            measured_fps = 30.0
            reopens = 0

            def read(self):
                frame = frames[min(counter["i"], len(frames) - 1)]
                counter["i"] += 1
                if counter["i"] >= len(frames):
                    counter["pipeline"]._stop.set()
                return frame, 1000.0 + counter["i"]

        pipeline = Pipeline(
            camera_index=0,
            width=4,
            height=4,
            fps=30,
            mirror=False,
            on_log=lambda _message: None,
            on_preview=lambda _frame: None,
            on_stats=lambda _stats: None,
            on_input=(seen.append if on_input else None),
        )
        pipeline.capture = FakeCapture()
        counter["pipeline"] = pipeline
        return pipeline

    def test_raw_camera_frames_reach_the_preview(self) -> None:
        seen: list[np.ndarray] = []
        self._pipeline(seen)._input_loop()
        self.assertTrue(seen, "the camera preview never received a frame")
        self.assertEqual(int(seen[0][0, 0, 0]), 10)  # untouched by the engine

    def test_preview_is_throttled_not_one_per_frame(self) -> None:
        """A 30 fps camera must not cost 30 GUI repaints a second."""
        seen: list[np.ndarray] = []
        pipeline = self._pipeline(seen)
        pipeline._input_loop()
        self.assertLessEqual(len(seen), 2, f"{len(seen)} previews for 3 frames")

    def test_no_preview_callback_is_allowed(self) -> None:
        pipeline = self._pipeline([], on_input=False)
        pipeline._input_loop()  # headless callers never ask for a preview
        self.assertEqual(pipeline.engine.frames_in, 3)


class CameraOpenFailureTests(unittest.TestCase):
    """DirectShow raises C++ exceptions on transient states; the app must not."""

    def test_open_capture_returns_a_closed_capture_instead_of_raising(self) -> None:
        from unittest import mock

        from mimiccam import capture

        with mock.patch.object(capture.cv2, "VideoCapture", side_effect=RuntimeError("boom")):
            camera = capture._open_capture(0)
        self.assertFalse(camera.isOpened())

    def test_the_camera_keeps_its_own_resolution_and_rate(self) -> None:
        """A device opened with numbers is a device told to change its mode."""
        from unittest import mock

        from mimiccam import capture

        asked: list[int] = []

        class _FakeCap:
            def isOpened(self) -> bool:  # noqa: N802 - cv2 API
                return True

            def set(self, prop: int, _value: float) -> bool:
                asked.append(prop)
                return True

            def release(self) -> None:
                return None

        with mock.patch.object(capture.cv2, "VideoCapture", return_value=_FakeCap()):
            capture._open_capture(0)

        for prop in (
            capture.cv2.CAP_PROP_FRAME_WIDTH,
            capture.cv2.CAP_PROP_FRAME_HEIGHT,
            capture.cv2.CAP_PROP_FPS,
        ):
            self.assertNotIn(prop, asked, f"requested camera property {prop}")

    def test_one_bad_index_does_not_kill_the_scan(self) -> None:
        from unittest import mock

        from mimiccam import capture

        def flaky(index, *_args):
            raise RuntimeError(f"camera {index} said no")

        with mock.patch.object(capture, "_open_capture", side_effect=flaky):
            self.assertEqual(capture.probe_cameras(3), [])

    def test_start_reports_failure_instead_of_crashing_the_window(self) -> None:
        from unittest import mock

        from mimiccam import capture

        camera = capture.CameraCapture(0, 640, 480, 30)
        with mock.patch.object(capture, "_open_capture", side_effect=RuntimeError("busy")):
            self.assertFalse(camera.start())


class VirtualCameraLoadFailureTests(unittest.TestCase):
    """A frozen build inside a deep folder cannot load the driver DLL.

    Windows reports "The filename or extension is too long". The app must start
    anyway and tell the user what to do, so this pins that message.
    """

    def setUp(self) -> None:
        from mimiccam import vcam

        self.vcam = vcam
        self._saved = (vcam.pyvirtualcam, vcam._IMPORT_ERROR)
        vcam.pyvirtualcam = None
        vcam._IMPORT_ERROR = "DLL load failed: The filename or extension is too long."

    def tearDown(self) -> None:
        self.vcam.pyvirtualcam, self.vcam._IMPORT_ERROR = self._saved

    def test_backend_probe_raises_actionable_error(self) -> None:
        with self.assertRaises(self.vcam.VirtualCameraError) as ctx:
            self.vcam.available_backends()
        message = str(ctx.exception)
        self.assertIn("too long", message)
        self.assertIn("C:\\MimicCam", message)

    def test_opening_camera_raises_instead_of_typeerror(self) -> None:
        camera = self.vcam.VirtualCamera()
        with self.assertRaises(self.vcam.VirtualCameraError):
            camera.open(1280, 720, 30)


if __name__ == "__main__":
    unittest.main()
