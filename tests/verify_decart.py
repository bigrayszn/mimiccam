# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Checks for the Lucy 2.5 engine.

Three things are verified here:

1. Reference-image normalisation. A character photo of any size becomes JPEG bytes
   padded to the output aspect, at least 512 px on the short side and at most
   1280 px on the long one, because Lucy under-uses a tiny or mis-shaped
   reference and the failure looks like "the swap does nothing".
2. Session plumbing. The engine really opens a WebRTC session against Decart's
   signalling server: local camera track created, model resolved, connect
   attempted, and the server's rejection surfaced as a readable log line instead
   of a hang or a traceback. Run it with a real key and the same code path
   completes; with a dummy key the server answers "Invalid API key", which is
   exactly the proof that everything before authentication works.
3. Whole-body substitution, measured on a live session. A still of a person
   whose clothes are unmistakably not the character's is pushed through Lucy; the
   output is scored on two things that separate a character swap from a face
   paste: does the output face match the character more than the source, and did
   the torso clothing change to the character's. This is the check that decides
   whether the prompt is right - a face-only result is a FAIL, not a pass.

Usage:
    python tests/verify_decart.py                       # plumbing, bogus key
    python tests/verify_decart.py --live                # session with DECART_API_KEY
    python tests/verify_decart.py --live --quality SOURCE.jpg [--notes "..."]
    python tests/verify_decart.py --live --quality SOURCE.jpg --old-prompt
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mimiccam import models, prompting, reference  # noqa: E402
from mimiccam.decart_engine import DecartEngine, load_reference_image  # noqa: E402

# The character this app was last set up with; --reference overrides it.
DEFAULT_REFERENCE = Path(
    r"C:\Users\offic\AppData\Roaming\MimicCam\characters"
    r"\WhatsApp Image 2026-09-15 at 1.17.18 PM.jpeg"
)
DEFAULT_NOTES = (
    "a black suit jacket, white shirt, black tie, black trousers, black leather shoes"
)
OLD_PROMPT = (
    "Substitute the character in the video with the person in the reference image. "
    "Keep their face, hair, skin tone, body shape and clothing consistent in every "
    "frame, with natural head movement and lighting that matches the room."
)


def check_reference_normalisation() -> bool:
    print("\n[1] reference image normalisation")
    with tempfile.TemporaryDirectory() as tmp:
        # Deliberately small and non-square: 240x160, like a cropped selfie.
        small = Path(tmp) / "small.png"
        cv2.imwrite(str(small), np.full((160, 240, 3), 128, dtype=np.uint8))

        data = load_reference_image(str(small))
        assert data is not None, "no bytes returned for a valid image"
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert array is not None, "returned bytes are not a decodable image"

        height, width = array.shape[:2]
        short_side = min(width, height)
        is_jpeg = data[:2] == b"\xff\xd8"
        print(f"    input 240x160 -> output {width}x{height}, {len(data)} bytes, jpeg={is_jpeg}")
        assert is_jpeg, "reference must be JPEG for the realtime API"
        assert short_side >= 512, f"short side {short_side} < 512, Lucy would under-use it"
        assert max(width, height) <= 1280, "reference should stay within 1280 px"

        # A missing file must be a clear error, not a silent empty prompt.
        try:
            load_reference_image(str(Path(tmp) / "nope.png"))
        except FileNotFoundError:
            print("    missing file -> FileNotFoundError (clear failure)")
        else:
            raise AssertionError("missing reference image did not raise")

    print("    PASS")
    return True


def check_session(key: str, seconds: float, live: bool) -> bool:
    print(f"\n[2] realtime session plumbing ({'live key' if live else 'dummy key'})")
    lines: list[str] = []
    lock = threading.Lock()

    def log(message: str) -> None:
        with lock:
            lines.append(message)
        print(f"    engine| {message}")

    engine = DecartEngine(
        api_key=key,
        width=1280,
        height=720,
        fps=30,
        prompt="Substitute the character in the video with the person in the reference image.",
        preset_key="character",
        reference_image=None,
        model="lucy-2.5",
        on_log=log,
    )
    engine.start()
    deadline = time.time() + seconds
    connected = False
    while time.time() < deadline:
        time.sleep(0.5)
        if engine.connected.is_set():
            connected = True
            break
        if live and engine.frames_published > 0:
            # Push one frame so a live session actually validates.
            engine.push(np.full((720, 1280, 3), 96, dtype=np.uint8), time.time())
    published = engine.frames_published
    received = engine.frames_received
    engine.stop()
    print(f"    published={published} received={received} connected={connected}")

    with lock:
        text = "\n".join(lines)
    assert "Connecting to lucy-2.5" in text, "engine never resolved the lucy-2.5 model"
    assert "1280x720" in text, "engine did not report the model's native resolution"

    if live:
        assert connected, f"live session never connected:\n{text}"
        print("    PASS - live Lucy 2.5 session connected")
        return True

    # Dummy key: the server must reject us in a readable way, and the engine must
    # have kept that failure out of the traceback path.
    rejected = any(
        needle in text.lower()
        for needle in ("invalid api key", "unauthorized", "decode", "token", "401", "403")
    )
    assert rejected, f"expected an auth rejection from Decart, got:\n{text}"
    print("    PASS - signalling reached Decart and auth rejection was handled cleanly")
    return True


def _letterbox(path: Path, size: tuple[int, int] = (1280, 720)) -> np.ndarray:
    """A still, at the shape and size of the realtime stream (BGR)."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(path))
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 26, np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def _detect(detector, bgr: np.ndarray):
    from mimiccam.face import largest

    return largest(detector.detect(bgr))


def _skin_in(rgb: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    return (cr >= 133) & (cr <= 180) & (cb >= 70) & (cb <= 130)


def _clothing_stats(bgr: np.ndarray, box: tuple[int, int, int, int]) -> Optional[dict]:
    """How much of the body below the chin is dark tailoring vs colour."""
    height, width = bgr.shape[:2]
    x, y, face_w, face_h = box
    y0 = min(height - 1, y + int(1.05 * face_h))
    y1 = min(height, y + int(2.6 * face_h))
    x0 = max(0, x - int(0.8 * face_w))
    x1 = min(width, x + int(1.8 * face_w))
    region = bgr[y0:y1, x0:x1]
    if region.size == 0:
        return None
    rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB).astype(np.int16)
    pixels = rgb.reshape(-1, 3)
    keep = ~_skin_in(region).reshape(-1)
    pixels = pixels[keep]
    if len(pixels) < 100:
        return None
    high = pixels.max(axis=1)
    low = pixels.min(axis=1)
    dark_neutral = float(np.mean((high < 95) & ((high - low) < 45)))
    colourful = float(np.mean((high - low) > 70))
    light_neutral = float(np.mean((low > 150) & ((high - low) < 45)))
    return {
        "dark": dark_neutral,
        "colourful": colourful,
        "light": light_neutral,
    }


def check_quality(
    key: str,
    source: Path,
    reference_path: Path,
    notes: str,
    seconds: float,
    old_prompt: bool,
    fast: bool,
) -> bool:
    """Drive a live session with a known performer and score the result."""
    from mimiccam.face import FaceDetector, FaceEmbedder

    print(f"\n[3] whole-body substitution on a live session ({seconds:.0f}s)")
    print(f"    source    {source.name}")
    print(f"    character {reference_path.name}")

    source_bgr = _letterbox(source)
    reference_bgr = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    detector = FaceDetector(models.path("detector"), refine=False)
    embedder = FaceEmbedder(models.path("identity"), providers=["CPUExecutionProvider"])

    source_face = _detect(detector, source_bgr)
    reference_face = _detect(detector, reference_bgr)
    if source_face is None or reference_face is None:
        print("    FAIL: no face found in the source or the character photo")
        return False
    reference_vec = embedder.embedding(reference_bgr, reference_face.landmarks)
    source_vec = embedder.embedding(source_bgr, source_face.landmarks)
    baseline = _clothing_stats(source_bgr, source_face.box)
    print(
        "    source clothing: dark {dark:.2f} colourful {colourful:.2f} light {light:.2f}".format(
            **(baseline or {"dark": -1, "colourful": -1, "light": -1})
        )
    )

    prompt = OLD_PROMPT if old_prompt else prompting.CHARACTER_TEMPLATE
    label = "old prompt (face-consistent)" if old_prompt else "whole-person prompt"
    print(f"    prompt: {label}")

    lines: list[str] = []
    lock = threading.Lock()

    def log(message: str) -> None:
        with lock:
            lines.append(message)

    engine = DecartEngine(
        api_key=key,
        width=1280,
        height=720,
        fps=30,
        prompt=prompt,
        # "custom" leaves the text exactly as written: with --old-prompt that is
        # the app as it shipped, no description and no whole-person template.
        preset_key="custom" if old_prompt else "character",
        reference_image=str(reference_path),
        character_notes="" if old_prompt else notes,
        model="lucy-2.5",
        fast_mode=fast,
        pause_when_idle=False,
        on_log=log,
    )
    print(f"    prompt sent: {engine.prompt}")

    engine.start()
    deadline = time.time() + 25.0
    while time.time() < deadline and not engine.connected.is_set():
        time.sleep(0.2)
    if not engine.connected.is_set():
        engine.stop()
        with lock:
            print("\n".join(lines))
        print("    FAIL: session never connected")
        return False

    frames: list[np.ndarray] = []
    last_ts = 0.0
    started = time.time()
    while time.time() - started < seconds:
        now = time.time()
        engine.push(cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB), now)
        frame, ts = engine.latest()
        if frame is not None and ts != last_ts and now - started > 2.0:
            last_ts = ts
            frames.append(frame)
            if len(frames) > 240:
                frames.pop(0)
        time.sleep(1.0 / 30.0)
    engine.stop()

    if len(frames) < 20:
        print(f"    FAIL: only {len(frames)} transformed frames came back")
        return False

    samples = frames[:: max(1, len(frames) // 24)]
    identity_ref: list[float] = []
    identity_src: list[float] = []
    clothing: list[dict] = []
    for frame in samples:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        face = _detect(detector, bgr)
        if face is None:
            continue
        vector = embedder.embedding(bgr, face.landmarks)
        identity_ref.append(float(np.dot(vector, reference_vec)))
        identity_src.append(float(np.dot(vector, source_vec)))
        stats = _clothing_stats(bgr, face.box)
        if stats:
            clothing.append(stats)

    if not identity_ref:
        print("    FAIL: no face in any transformed frame")
        return False

    face_rate = len(identity_ref) / len(samples)
    to_ref = float(np.mean(identity_ref))
    to_src = float(np.mean(identity_src))
    dark = float(np.mean([item["dark"] for item in clothing])) if clothing else -1.0
    colourful = float(np.mean([item["colourful"] for item in clothing])) if clothing else -1.0
    light = float(np.mean([item["light"] for item in clothing])) if clothing else -1.0

    print(f"    frames {len(frames)}, face in {face_rate * 100:.0f}% of samples")
    print(f"    face similarity: character {to_ref:.3f} / source {to_src:.3f}")
    print(
        "    output clothing: dark {:.2f} colourful {:.2f} light {:.2f}".format(
            dark, colourful, light
        )
    )

    out_dir = Path(__file__).resolve().parent / "captures"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    saved = []
    for index in (len(samples) // 4, len(samples) // 2, 3 * len(samples) // 4):
        name = out_dir / f"lucy_{stamp}_{index}.jpg"
        cv2.imwrite(str(name), cv2.cvtColor(samples[index], cv2.COLOR_RGB2BGR))
        saved.append(name.name)
    print(f"    frames saved: {', '.join(saved)}")

    identity_ok = to_ref > to_src
    baseline_dark = (baseline or {}).get("dark", 0.0)
    baseline_colour = (baseline or {}).get("colourful", 0.0)
    clothing_ok = (dark - baseline_dark) > 0.25 or (baseline_colour - colourful) > 0.25
    print(f"    identity moved to the character: {'YES' if identity_ok else 'NO'}")
    print(f"    clothing replaced:              {'YES' if clothing_ok else 'NO'}")

    ok = face_rate > 0.75 and identity_ok and clothing_ok
    print("    PASS" if ok else "    FAIL - this is still a face swap, not a character swap")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="use DECART_API_KEY for a real session")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--key", default="")
    parser.add_argument("--quality", type=Path, default=None,
                        help="still of a performer to push through a live session")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--notes", default=DEFAULT_NOTES)
    parser.add_argument("--old-prompt", action="store_true",
                        help="run the previous face-consistent prompt, for comparison")
    parser.add_argument("--fast", action="store_true", help="billed at 2x")
    args = parser.parse_args()

    key = args.key or os.environ.get("DECART_API_KEY", "").strip()
    if args.live and not key:
        print("FAIL: --live needs DECART_API_KEY or --key")
        return 2
    if not key:
        key = "ek_dummy_key_for_plumbing_check"

    ok = check_reference_normalisation()
    if args.quality is not None:
        if not args.live:
            print("FAIL: --quality needs --live (it spends a real session)")
            return 2
        ok = check_quality(
            key,
            args.quality,
            args.reference,
            args.notes,
            min(args.seconds, 20.0),
            args.old_prompt,
            args.fast,
        ) and ok
    else:
        ok = check_session(key, args.seconds, args.live) and ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
