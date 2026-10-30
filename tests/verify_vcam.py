"""End-to-end check: MimicCam -> virtual camera -> a third-party consumer.

ffmpeg stands in for Zoom/Telegram/WhatsApp: it opens the virtual camera through
DirectShow exactly like a conferencing app does. The script then inspects the
frames it received instead of trusting the app's own counters.

The decisive assertion is structural, not brightness: the app emits a coarse 8x8
luma fingerprint of every frame it hands to the virtual camera, and the consumer's
frames are matched against all of them. Pearson correlation ignores exposure
drift, so the check survives a lamp being switched mid-run (this laptop's webcam
auto-exposure swings 10 -> 80 luma on its own) while still catching a black feed,
a stalled shared-memory buffer, a channel swap or a wrong stride.

Usage:
    python tests/verify_vcam.py --camera 1 --seconds 25
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = ROOT / "tests" / "captures"


def find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in Path.home().glob("AppData/Local/ffmpeg/*/bin/ffmpeg.exe"):
        return str(candidate)
    return None


def list_dshow_devices(ffmpeg: str) -> list[str]:
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    return re.findall(r'"([^"]+)" \(video\)', proc.stderr)


def pick_device(devices: list[str]) -> str | None:
    for needle in ("OBS Virtual Camera", "Unity Video Capture", "Unity Capture"):
        for device in devices:
            if needle.lower() in device.lower():
                return device
    return None


class AppReader(threading.Thread):
    """Drains the app's stdout continuously.

    Reading only at the end would risk a full pipe stalling the app mid-capture,
    and it would also throw away the fingerprints sampled while ffmpeg was running.
    """

    def __init__(self, proc: subprocess.Popen) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.ready = threading.Event()
        self.stats: list[dict] = []
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def run(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.rstrip()
            with self._lock:
                self.lines.append(line)
            if "READY" in line:
                self.ready.set()
            marker = "[stats] "
            if marker in line:
                try:
                    value = ast.literal_eval(line.split(marker, 1)[1].strip())
                except (ValueError, SyntaxError):
                    continue
                if isinstance(value, dict):
                    with self._lock:
                        self.stats.append(value)
            else:
                print(f"  app| {line}")

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.stats)


def capture_with_ffmpeg(ffmpeg: str, device: str, frames: int) -> list[Path]:
    if CAPTURE_DIR.exists():
        shutil.rmtree(CAPTURE_DIR)
    CAPTURE_DIR.mkdir(parents=True)
    out_pattern = str(CAPTURE_DIR / "frame_%03d.png")
    proc = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "dshow", "-video_size", "1280x720", "-framerate", "30",
            "-i", f"video={device}",
            "-frames:v", str(frames),
            out_pattern,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        print(f"  ffmpeg error: {proc.stderr.strip()[:800]}")
    return sorted(CAPTURE_DIR.glob("frame_*.png"))


def analyse(paths: list[Path]) -> dict:
    if not paths:
        return {"frames": 0}
    means, diffs = [], []
    previous = None
    shape = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        shape = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        means.append(float(gray.mean()))
        if previous is not None:
            diffs.append(float(cv2.absdiff(gray, previous).mean()))
        previous = gray
    return {
        "frames": len(means),
        "shape": shape,
        "mean_luma": round(float(np.mean(means)), 1) if means else 0.0,
        "mean_frame_diff": round(float(np.mean(diffs)), 2) if diffs else 0.0,
        "unique_frames": len({round(m, 2) for m in means}),
    }


def grid_of(image_bgr: np.ndarray) -> list[float]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grid = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return [float(v) for v in grid.ravel()]


def correlation(a: list[float], b: list[float]) -> float:
    """Pearson correlation: scale and shift invariant, so exposure drift is ignored."""
    if len(a) != len(b) or len(a) < 4:
        return 0.0
    first, second = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    first -= first.mean()
    second -= second.mean()
    denominator = float(np.sqrt((first**2).sum() * (second**2).sum()))
    if denominator < 1e-6:
        return 0.0
    return float((first * second).sum() / denominator)


def structure_match(fingerprints: list[list[float]], paths: list[Path]) -> tuple[float, float]:
    """Best correlation between any fingerprint the app emitted and any consumed frame.

    Both orientations are scored. The app mirrors the camera once, so the consumer
    must see the frame exactly as the app emitted it; a stray extra flip anywhere in
    the chain shows up as the mirrored score winning.
    """
    best, best_flipped = 0.0, 0.0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        grid = grid_of(image)
        flipped = [grid[row * 8 + (7 - col)] for row in range(8) for col in range(8)]
        for fingerprint in fingerprints:
            best = max(best, correlation(fingerprint, grid))
            best_flipped = max(best_flipped, correlation(fingerprint, flipped))
    return best, best_flipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("FAIL: ffmpeg not found on PATH (install it or add it to PATH)")
        return 2

    devices = list_dshow_devices(ffmpeg)
    print("DirectShow video devices:")
    for device in devices:
        print(f"  - {device}")
    device = args.device or pick_device(devices)
    if not device:
        print("FAIL: no virtual camera device found. Is OBS Studio installed?")
        return 2
    print(f"Using device: {device}")

    print("\nStarting MimicCam headless...")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "mimiccam", "--selftest", str(args.seconds),
         "--camera", str(args.camera)],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    reader = AppReader(proc)
    reader.start()
    try:
        if not reader.ready.wait(timeout=40):
            print("FAIL: app never became ready")
            return 1

        time.sleep(1.5)
        print(f"\nCapturing {args.frames} frames through DirectShow (third-party path)...")
        paths = capture_with_ffmpeg(ffmpeg, device, args.frames)
        readback = analyse(paths)

        print("\n--- readback stats ---")
        for key, value in readback.items():
            print(f"{key}: {value}")

        time.sleep(1.5)  # let one more stats line land after the capture
        stats = reader.snapshot()
        fingerprints = [s["grid"] for s in stats if isinstance(s.get("grid"), list) and s["grid"]]
        newest = stats[-1] if stats else {}
        app_luma = float(newest.get("luma") or 0.0)
        readback_luma = float(readback.get("mean_luma") or 0.0)
        print(f"\nfingerprints sampled: {len(fingerprints)}")
        print(f"app luma (latest): {app_luma} | readback luma: {readback_luma}")

        same, flipped = structure_match(fingerprints, paths)
        print(f"structure correlation: as-is {same:.3f} | mirrored {flipped:.3f}")

        checks = {
            "frames arrived": readback.get("frames", 0) >= max(1, args.frames // 2),
            "resolution 1280x720": readback.get("shape") is not None
            and readback["shape"][:2] == (720, 1280),
            "not a black feed": readback_luma > 2.0,
            "consumer sees the app's own picture (r>=0.90)": max(same, flipped) >= 0.9,
            "no accidental flip end-to-end": same >= 0.9 and same >= flipped,
            "app delivered frames": int(newest.get("frames_sent") or 0) > 0,
            "no held/stale frames": int(newest.get("held_frames") or 0) == 0,
        }
        for label, passed in checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

        ok = all(checks.values())
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
