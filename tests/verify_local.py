# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""End-to-end check of the free, offline engine.

This proves the no-credits path with numbers instead of hope:

1. an animated portrait stands in for a webcam (no face in front of the lens
   required, and the motion is identical on every run),
2. its identity is swapped for a second, different person from a photo,
3. the result goes through the real virtual camera,
4. ffmpeg opens that camera through DirectShow exactly like Zoom does and the
   frames it receives are measured.

The decisive number is identity similarity: the ArcFace embedding of the face
*ffmpeg received* is compared with the character photo. A swap that only changed
pixels, or a camera that delivered stale frames, fails that check.

The two portraits are official White House photographs (public domain), fetched
once into a temp folder:
  reference  Barack Obama, 2012 official portrait (Pete Souza)
  target     Joe Biden, 2021 official portrait (Adam Schultz)

    python tests/verify_local.py --seconds 8

``--detail`` runs the same take with the GFPGAN detail pass on. On an idle
machine both hit the loop's 30 fps ceiling and the pass costs 2-3 ms a frame;
repeat the run on a loaded machine before reading anything into the numbers.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from mimiccam import face, models  # noqa: E402
from mimiccam.local_engine import LocalEngine  # noqa: E402
from mimiccam.vcam import VirtualCamera  # noqa: E402
from verify_vcam import (  # noqa: E402
    capture_with_ffmpeg,
    find_ffmpeg,
    list_dshow_devices,
    pick_device,
)

FIXTURES = {
    "reference": (
        "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
        "obama.jpg",
    ),
    "target": (
        "https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg",
        "biden.jpg",
    ),
}


def fixture(kind: str, override: str = "") -> Path:
    if override:
        return Path(override)
    url, name = FIXTURES[kind]
    cache = Path.home() / "AppData" / "Local" / "Temp" / "mimiccam-verify"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists() or path.stat().st_size < 100_000:
        print(f"fetching {kind} photo from Wikimedia Commons...")
        request = urllib.request.Request(url, headers={"User-Agent": "MimicCam/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    return path


def webcam_framing(photo_bgr: np.ndarray, detector: face.FaceDetector) -> np.ndarray:
    """Crop a portrait to a 1280x720 webcam view with the face in the middle."""
    height, width = photo_bgr.shape[:2]
    found = face.largest(detector.detect(photo_bgr))
    if found is None:
        raise SystemExit("no face in the target photo")
    cx, cy = found.box[0] + found.box[2] // 2, found.box[1] + found.box[3] // 2
    side = int(found.box[3] * 5)
    y0 = max(0, min(height - side, cy - side // 2))
    x0 = max(0, min(width - side, cx - side // 2))
    return cv2.resize(photo_bgr[y0:y0 + side, x0:x0 + side], (1280, 720),
                      interpolation=cv2.INTER_AREA)


def animate(base_bgr: np.ndarray, count: int) -> list[np.ndarray]:
    """Small pan, drift and roll: enough motion to catch a frozen texture."""
    frames = []
    for index in range(count):
        phase = (index / max(1, count - 1)) * 2 * math.pi
        matrix = cv2.getRotationMatrix2D((640, 360), 1.5 * math.sin(phase * 0.7), 1.0)
        matrix[0, 2] += 25 * math.sin(phase)
        matrix[1, 2] += 12 * math.cos(phase * 1.3)
        frames.append(
            cv2.warpAffine(base_bgr, matrix, (1280, 720), borderMode=cv2.BORDER_REFLECT101)
        )
    return frames


def identity_of(frame_bgr: np.ndarray, detector: face.FaceDetector,
                embedder: face.FaceEmbedder, reference: np.ndarray) -> float | None:
    found = face.largest(detector.detect(frame_bgr))
    if found is None:
        return None
    return float(embedder.embedding(frame_bgr, found.landmarks) @ reference)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="", help="character photo (who you become)")
    parser.add_argument("--target", default="", help="photo standing in for the webcam")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--device", default="", help="virtual camera device name")
    parser.add_argument("--capture", type=int, default=25, help="frames to read back")
    parser.add_argument("--skip-vcam", action="store_true")
    parser.add_argument("--studio", action="store_true",
                        help="also seat the person in the reference photo's room")
    parser.add_argument("--detail", action="store_true",
                        help="run the GFPGAN detail pass the engine now ships on")
    args = parser.parse_args()

    absent = models.missing()
    if absent:
        print(models.describe_missing(absent))
        print("Run the app once to download them, or: python -c "
              "\"from mimiccam import models; models.ensure()\"")
        return 2

    detector = face.FaceDetector(models.path("detector"))
    embedder = face.FaceEmbedder(models.path("identity"))

    reference_path = fixture("reference", args.reference)
    target_path = fixture("target", args.target)
    reference_photo = face.load_image(reference_path)
    target_photo = face.load_image(target_path)
    reference_face = face.largest(detector.detect(reference_photo))
    target_face = face.largest(detector.detect(target_photo))
    if reference_face is None or target_face is None:
        print("FAIL: a fixture photo has no detectable face")
        return 2
    character = embedder.embedding(reference_photo, reference_face.landmarks)
    original = embedder.embedding(target_photo, target_face.landmarks)
    print(f"reference face: {reference_face.box}  target face: {target_face.box}")
    print(f"two identities are different (cosine {float(character @ original):+.3f})")

    base = webcam_framing(target_photo, detector)
    frames_needed = int(args.seconds * 30)
    frames = animate(base, frames_needed)
    before = identity_of(base, detector, embedder, character)
    print(f"before swap, the 'camera' looks like the target (cosine {before:+.3f})")

    engine = LocalEngine(
        reference_image=str(reference_path),
        preset_key="character",
        width=1280,
        height=720,
        fps=30,
        studio_background=args.studio,
        studio_image=str(reference_path) if args.studio else None,
        on_log=lambda message: print(f"  engine| {message}"),
        detail=args.detail,
    )

    camera = None
    captures: list[Path] = []
    capture_thread: threading.Thread | None = None
    ffmpeg = find_ffmpeg()

    if not args.skip_vcam:
        if not ffmpeg:
            print("FAIL: ffmpeg not on PATH, needed for the readback")
            return 2
        devices = list_dshow_devices(ffmpeg)
        device = args.device or pick_device(devices)
        if not device:
            print("FAIL: no virtual camera found. Install OBS Studio first.")
            return 2
        print(f"readback device: {device}")

    started = time.perf_counter()
    engine.start()
    print(f"engine ready in {time.perf_counter() - started:.2f}s")

    if not args.skip_vcam:
        camera = VirtualCamera(on_log=lambda message: print(f"  vcam| {message}"))
        camera.open(1280, 720, 30, "")
        print(f"virtual camera: {camera.device}")

    outputs: list[np.ndarray] = []
    stamps: set[float] = set()
    loop_start = time.perf_counter()
    period = 1.0 / 30.0
    deadline = time.perf_counter()
    try:
        for index, frame in enumerate(frames):
            if camera is not None and index == 15 and ffmpeg:
                capture_thread = threading.Thread(
                    target=lambda: captures.extend(
                        capture_with_ffmpeg(ffmpeg, device, args.capture)
                    ),
                    daemon=True,
                )
                capture_thread.start()
            engine.push(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), time.perf_counter())
            deadline += period
            sleep_for = deadline - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                deadline = time.perf_counter()

            produced, stamp = engine.latest()
            if produced is not None:
                stamps.add(round(stamp, 4))
                # Only convert a few frames for measurement: doing it every
                # iteration would put the harness, not the engine, under load.
                if len(stamps) % 5 == 0:
                    outputs.append(cv2.cvtColor(produced, cv2.COLOR_RGB2BGR))
                if camera is not None:
                    camera.send(produced)
        if capture_thread is not None:
            capture_thread.join(timeout=30)
    finally:
        if camera is not None:
            camera.close()
        engine.stop()

    wall = time.perf_counter() - loop_start
    # Distinct engine timestamps, not loop iterations: the harness samples at
    # 30 Hz and the engine produces at its own rate, so counting pixels would
    # blame the engine for the harness repeating the newest frame.
    delivered = engine.measured_fps

    print("\n--- engine ---")
    print(f"engine loop               {delivered:.1f} fps "
          f"({len(stamps)} distinct frames in {wall:.1f}s)")
    print(f"frames measured           {len(outputs)}")
    print(f"identity refreshes        {engine.texture_refreshes} "
          f"({engine.render_fps:.1f}/s)")
    print(f"frame-to-frame latency    {engine.latency_ms():.0f} ms")
    print(f"frames without a face     {engine.frames_no_face}")

    recent = outputs[-12:] if outputs else []
    sims = [s for s in (identity_of(o, detector, embedder, character) for o in recent)
            if s is not None]
    mean_sim = float(np.mean(sims)) if sims else 0.0

    diffs = [
        float(cv2.absdiff(cv2.cvtColor(outputs[i], cv2.COLOR_BGR2GRAY),
                          cv2.cvtColor(outputs[i - 1], cv2.COLOR_BGR2GRAY)).mean())
        for i in range(1, len(outputs))
    ]
    moving = sum(1 for d in diffs if d > 0.05)

    print("\n--- swapped identity (engine output) ---")
    print(f"similarity to character   {mean_sim:+.3f} (was {before:+.3f} as the target)")

    readback_ok = True
    readback_sim: list[float] = []
    if not args.skip_vcam:
        print("\n--- readback through the virtual camera (ffmpeg as Zoom) ---")
        shape = None
        luma = 0.0
        for path in captures:
            image = cv2.imread(str(path))
            if image is None:
                continue
            shape = image.shape
            luma = max(luma, float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean()))
            value = identity_of(image, detector, embedder, character)
            if value is not None:
                readback_sim.append(value)
        mean_readback = float(np.mean(readback_sim)) if readback_sim else 0.0
        print(f"frames received           {len(captures)}")
        print(f"resolution                {shape[1]}x{shape[0]}" if shape else "resolution -")
        print(f"brightest frame luma      {luma:.1f}")
        print(f"faces measured            {len(readback_sim)}")
        print(f"similarity to character   {mean_readback:+.3f}")
        readback_ok = (
            len(captures) >= max(5, args.capture // 2)
            and shape is not None
            and shape[:2] == (720, 1280)
            and luma > 2.0
            and mean_readback > 0.25
        )

    checks = {
        "models present": not absent,
        "engine keeps up (>=25 fps)": delivered >= 25.0,
        "identity refreshes (>=6/s)": engine.render_fps >= 6.0,
        "latency under 150 ms": engine.latency_ms() < 150.0,
        "every frame has a face": engine.frames_no_face == 0,
        "swap changes identity (>=0.30)": mean_sim >= 0.30,
        "frames keep changing": moving >= max(1, int(len(diffs) * 0.5)),
    }
    if args.studio:
        # The room composite and the person matte both run every frame, so the
        # studio path is slower than a plain swap and has its own bar; see the
        # README for the measured numbers. 20 fps is the floor for a subject
        # who fills the frame (measured 23-25); a normal desk-webcam frame runs
        # 16-25 ms and is limited by the camera instead.
        checks.pop("engine keeps up (>=25 fps)", None)
        checks["studio room stayed live (>=20 fps)"] = delivered >= 20.0
    if not args.skip_vcam:
        checks["readback shows the character"] = readback_ok

    print()
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    ok = all(checks.values())
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
