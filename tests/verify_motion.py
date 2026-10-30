"""Motion check for the free engine: does the swap stay glued to the performer?

``tests/verify_local.py`` answers "is this the right person?" on a still frame.
It cannot answer "does the face stay on the head while the head moves?", which is
the complaint a still-frame test never catches. This harness drives the real
engine with a scripted performer whose motion is known exactly:

* **detector error** - the face YuNet finds in each *input* frame versus where the
  scripted transform actually put the real face. Proves the tracker follows the
  performer rather than drifting off.
* **glue error** - the landmarks the engine pasted with versus the face it saw in
  that same frame. This is the number that matters: the character's face has a
  different geometry from the subject's, so comparing output landmarks with the
  subject's would measure face *shape*, not tracking. The paste alignment is
  compared instead.
* **jitter** - the spread of the glue error. In a still take this is the swap
  swimming on its own.
* **drift** - the slope of the glue error over a still take: a swap that slowly
  slides off the head.
* **rigidity** - consecutive output frames, warped by the known motion, compared
  in the head box. Glued content leaves camera noise; content that stays put
  while the head moves leaves a large residual.
* **dropout** - the face is covered for 450 ms, as a hand or a blink would. The
  character must stay on screen through the 400 ms pose hold and the frame must
  not flash back to the raw camera.

Takes: still, pan, zoom, roll, then the dropout. Deterministic, so a regression
is a number rather than a feeling.

    python tests/verify_motion.py
    python tests/verify_motion.py --frames 60 --save-frames
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from mimiccam import face, models  # noqa: E402
from mimiccam.local_engine import DETECT_MAX_SIDE, LocalEngine  # noqa: E402
from verify_local import fixture, identity_of, webcam_framing  # noqa: E402

WIDTH, HEIGHT = 1280, 720
CENTER = (WIDTH / 2.0, HEIGHT / 2.0)
MOTION_TAKES = ("still", "pan", "zoom", "roll")


def transform_for(take: str, index: int, count: int) -> np.ndarray:
    """The known camera motion for one frame of a take."""
    phase = (index / max(1, count - 1)) * 2.0 * math.pi
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    if take == "still":
        return matrix
    if take == "pan":
        matrix[0, 2] = 70.0 * math.sin(phase)
        matrix[1, 2] = 18.0 * math.sin(phase * 0.6)
        return matrix
    if take == "zoom":
        return cv2.getRotationMatrix2D(CENTER, 0.0, 1.0 + 0.09 * math.sin(phase * 0.8))
    if take == "roll":
        return cv2.getRotationMatrix2D(CENTER, 6.0 * math.sin(phase), 1.0)
    raise ValueError(take)


def face_block(base: np.ndarray, landmarks: np.ndarray) -> tuple[int, int, int, int, np.ndarray]:
    """Where to put the flat block that hides the face, and the block itself."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
    span = float(np.linalg.norm(points[0] - points[1]))
    cx = float((points[0][0] + points[1][0]) / 2.0)
    cy = float((points[0][1] + points[1][1]) / 2.0)
    x0, x1 = max(0, int(cx - span * 1.35)), min(WIDTH, int(cx + span * 1.35))
    y0, y1 = max(0, int(cy - span * 1.8)), min(HEIGHT, int(cy + span * 1.8))
    # A flat grey block: YuNet has nothing to lock onto, like a hand over the
    # lens or a head turned away.
    return x0, y0, x1, y1, np.full((y1 - y0, x1 - x0, 3), 96, dtype=np.uint8)


class Take:
    """One take: pushed frames, the face in each, engine output and its paste."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.frames: list[np.ndarray] = []
        self.truth: list[np.ndarray] = []
        self.detected: list[np.ndarray | None] = []
        self.outputs: list[np.ndarray] = []
        self.pasted: list[np.ndarray | None] = []
        self.times: list[float] = []


def pump(engine: LocalEngine, take: Take, frame: np.ndarray, truth: np.ndarray,
         detector: face.FaceDetector, last_stamp: float) -> float:
    """Push one frame, return the output the engine made from it.

    One frame in, one frame out: the engine is allowed to be slower than the
    scripted 30 fps, but never to skip the frame under test. That keeps the
    per-frame motion small and the paste/truth pair unambiguous.
    """
    take.frames.append(frame)
    take.truth.append(truth)
    found = face.largest(detector.detect(frame))
    take.detected.append(found.landmarks if found is not None else None)

    engine.push(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), time.perf_counter())
    deadline = time.perf_counter() + 1.0
    produced = None
    stamp = last_stamp
    while time.perf_counter() < deadline:
        produced, stamp = engine.latest()
        if produced is not None and stamp > last_stamp:
            break
        time.sleep(0.002)
    if produced is None or stamp <= last_stamp:
        take.outputs.append(frame)
        take.pasted.append(None)
        take.times.append(time.perf_counter())
        return last_stamp

    take.outputs.append(cv2.cvtColor(produced, cv2.COLOR_RGB2BGR))
    pasted = engine.paste_landmarks
    take.pasted.append(pasted)
    take.times.append(time.perf_counter())
    return stamp


def run_motion_take(engine: LocalEngine, detector: face.FaceDetector, name: str,
                    count: int) -> Take:
    take = Take(name)
    last_stamp = 0.0
    for index in range(count):
        matrix = transform_for(name, index, count)
        # Constant borders: a reflected border clones the face at the frame edge
        # and the detector then has two identical faces to choose between.
        frame = cv2.warpAffine(webcam_base, matrix, (WIDTH, HEIGHT),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(24, 24, 24))
        truth = cv2.transform(base_landmarks.reshape(1, -1, 2), matrix).reshape(-1, 2)
        last_stamp = pump(engine, take, frame, truth, detector, last_stamp)
    return take


def run_dropout_take(engine: LocalEngine, detector: face.FaceDetector,
                     cover_seconds: float = 0.45, total_seconds: float = 1.4) -> Take:
    """Cover the face in wall-clock time, so the 400 ms hold is 400 ms."""
    take = Take("dropout")
    x0, y0, x1, y1, block = face_block(webcam_base, base_landmarks)
    last_stamp = 0.0
    started = time.perf_counter()
    while time.perf_counter() - started < total_seconds:
        covered = (time.perf_counter() - started) < cover_seconds
        frame = webcam_base.copy()
        if covered:
            frame[y0:y1, x0:x1] = block
        last_stamp = pump(engine, take, frame, base_landmarks.copy(), detector, last_stamp)
        take.covered = getattr(take, "covered", [])
        take.covered.append(covered)
    return take


def glue_errors(take: Take) -> tuple[list[float], list[float]]:
    """Paste-versus-detector error per frame: px, and % of the eye distance."""
    px: list[float] = []
    pct: list[float] = []
    for pasted, got in zip(take.pasted, take.detected):
        if pasted is None or got is None:
            continue
        span = float(np.linalg.norm(got[0] - got[1])) or 1.0
        distance = float(np.mean(np.linalg.norm(pasted - got, axis=1)))
        px.append(distance)
        pct.append(100.0 * distance / span)
    return px, pct


def detector_errors(take: Take) -> tuple[list[float], list[float]]:
    """Detector-versus-scripted-truth error: does tracking follow the performer?"""
    px: list[float] = []
    pct: list[float] = []
    for got, want in zip(take.detected, take.truth):
        if got is None:
            continue
        span = float(np.linalg.norm(want[0] - want[1])) or 1.0
        distance = float(np.mean(np.linalg.norm(got - want, axis=1)))
        px.append(distance)
        pct.append(100.0 * distance / span)
    return px, pct


def slip_vectors(take: Take) -> np.ndarray:
    rows = [np.mean(pasted - got, axis=0)
            for pasted, got in zip(take.pasted, take.detected)
            if pasted is not None and got is not None]
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 2), np.float32)


def sustained_slip(slip: np.ndarray, window: int = 5) -> float:
    """The low-frequency part of the slip: what the eye reads as sliding.

    Per-frame paste-vs-detection distance also contains YuNet's own landmark
    noise (5-10 px on a resampled frame). That noise is not visible: the paste
    follows a smoothed path, so what a person notices is the slow part - lag on
    a pan, creep on a still. Averaging the slip over a few frames separates the
    two.
    """
    if len(slip) < window:
        return 0.0 if len(slip) == 0 else float(np.linalg.norm(slip.mean(axis=0)))
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.stack([np.convolve(slip[:, axis], kernel, mode="valid")
                         for axis in range(2)], axis=1)
    return float(np.max(np.linalg.norm(smoothed, axis=1)))


def drift(slip: np.ndarray, seconds: float) -> float:
    """Slope of the slip over a take, in px per second."""
    if len(slip) < 4 or seconds <= 0:
        return 0.0
    times = np.linspace(0.0, seconds, len(slip))
    slopes = [float(np.polyfit(times, slip[:, axis], 1)[0]) for axis in range(2)]
    return float(np.hypot(*slopes))


def rigidity(take: Take) -> float | None:
    """Output travel versus input travel, both warped by the known motion.

    A value near 1 means the swap moves exactly with the head (camera noise is
    both numerator and denominator). A large value means content stayed put.
    """
    if len(take.frames) < 3 or take.name not in MOTION_TAKES:
        return None

    def residual(frames: list[np.ndarray]) -> float:
        values: list[float] = []
        for index in range(1, len(frames)):
            undo = np.linalg.inv(np.vstack([
                transform_for(take.name, index - 1, len(frames)), [0.0, 0.0, 1.0]
            ]))[:2]
            warped = cv2.warpAffine(frames[index], undo, (WIDTH, HEIGHT),
                                    flags=cv2.INTER_LINEAR)
            want = take.truth[index]
            cx, cy = float(want[:, 0].mean()), float(want[:, 1].mean())
            span = float(np.linalg.norm(want[0] - want[1])) or 1.0
            half = int(span * 1.4)
            x0, x1 = max(0, int(cx - half)), min(WIDTH, int(cx + half))
            y0, y1 = max(0, int(cy - half)), min(HEIGHT, int(cy + half))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            diff = cv2.absdiff(cv2.cvtColor(warped[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(frames[index - 1][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
            values.append(float(diff.mean()))
        return float(np.mean(values)) if values else 0.0

    baseline = residual(take.frames)
    if baseline < 0.15:  # a still take against a still camera: nothing to divide
        return None
    return residual(take.outputs) / baseline


def best_lag(take: Take) -> tuple[int, float]:
    """Frames of delay between the paste and the face it is following."""
    best = (0, float("inf"))
    for lag in range(0, 4):
        values = []
        for index in range(lag, len(take.pasted)):
            pasted = take.pasted[index]
            got = take.detected[index - lag]
            if pasted is None or got is None:
                continue
            values.append(float(np.mean(np.linalg.norm(pasted - got, axis=1))))
        if values and float(np.mean(values)) < best[1]:
            best = (lag, float(np.mean(values)))
    return best


def main() -> int:
    global webcam_base, base_landmarks

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="", help="character photo")
    parser.add_argument("--target", default="", help="photo standing in for the webcam")
    parser.add_argument("--frames", type=int, default=48, help="frames per motion take")
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()

    absent = models.missing()
    if absent:
        print(models.describe_missing(absent))
        return 2

    # The same detection size the engine uses (DETECT_MAX_SIDE): YuNet returns
    # landmarks that differ by tens of pixels when its input resolution changes,
    # and this harness compares its own detection with the engine's paste.
    # Same configuration as the engine's detector (coarse pass, no crop refine),
    # otherwise this compares two different landmark fits instead of paste
    # against detection.
    detector = face.FaceDetector(models.path("detector"), max_side=DETECT_MAX_SIDE,
                                 refine=False)
    embedder = face.FaceEmbedder(models.path("identity"))

    reference_path = fixture("reference", args.reference)
    target_path = fixture("target", args.target)
    reference_photo = face.load_image(reference_path)
    target_photo = face.load_image(target_path)
    reference_face = face.largest(detector.detect(reference_photo))
    if reference_face is None:
        print("FAIL: no face in the character photo")
        return 2
    character = embedder.embedding(reference_photo, reference_face.landmarks)

    webcam_base = webcam_framing(target_photo, detector)
    base_face = face.largest(detector.detect(webcam_base))
    if base_face is None:
        print("FAIL: no face in the webcam framing")
        return 2
    base_landmarks = base_face.landmarks

    engine = LocalEngine(
        reference_image=str(reference_path),
        preset_key="character",
        width=WIDTH,
        height=HEIGHT,
        fps=30,
        on_log=lambda message: print(f"  engine| {message}"),
    )
    started = time.perf_counter()
    engine.start()
    print(f"engine ready in {time.perf_counter() - started:.2f}s")

    takes: list[Take] = []
    dropout: Take | None = None
    try:
        for name in MOTION_TAKES:
            takes.append(run_motion_take(engine, detector, name, args.frames))
        dropout = run_dropout_take(engine, detector)
    finally:
        engine.stop()

    print(f"\nframes without a face    {engine.frames_no_face}")
    print(f"identity refreshes       {engine.texture_refreshes} ({engine.render_fps:.1f}/s)")
    print(f"frame latency            {engine.latency_ms():.0f} ms")

    print("\n--- tracking ---")
    print(f"{'take':8} {'glue px':>8} {'glue p95':>9} {'glue %':>7} "
          f"{'det px':>7} {'det p95':>8} {'jitter':>7} {'slide':>6} {'drift/s':>8} "
          f"{'rigid':>6} {'lag':>4}")
    rows: dict[str, dict[str, float]] = {}
    for take in takes:
        gpx, gpct = glue_errors(take)
        dpx, dpct = detector_errors(take)
        slip = slip_vectors(take)
        jitter = float(np.sqrt((slip ** 2).sum(axis=1).mean())) if len(slip) else 999.0
        lag, _ = best_lag(take)
        seconds = max(1e-3, take.times[-1] - take.times[0]) if len(take.times) > 1 else 0.0
        rigid = rigidity(take)
        rows[take.name] = {
            "glue": float(np.mean(gpx)) if gpx else 999.0,
            "glue_p95": float(np.percentile(gpx, 95)) if gpx else 999.0,
            "glue_pct": float(np.mean(gpct)) if gpct else 999.0,
            "glue_p95_pct": float(np.percentile(gpct, 95)) if gpct else 999.0,
            "det": float(np.mean(dpx)) if dpx else 999.0,
            "det_p95": float(np.percentile(dpx, 95)) if dpx else 999.0,
            "jitter": jitter,
            "slide": sustained_slip(slip),
            "drift": drift(slip, seconds),
            "rigid": rigid if rigid is not None else float("nan"),
            "lag": lag,
            "frames": len(take.frames),
            "pasted": sum(1 for p in take.pasted if p is not None),
        }
        row = rows[take.name]
        rigid_text = "n/a" if rigid is None else f"{rigid:.2f}"
        print(f"{take.name:8} {row['glue']:8.2f} {row['glue_p95']:9.2f} {row['glue_pct']:7.2f} "
              f"{row['det']:7.2f} {row['det_p95']:8.2f} {jitter:7.2f} {row['slide']:6.2f} "
              f"{row['drift']:8.3f} {rigid_text:>6} {lag:4d}")

    print("\n--- identity while moving ---")
    identity: dict[str, float] = {}
    for take in takes:
        samples = [s for s in (identity_of(out, detector, embedder, character)
                               for out in take.outputs[::3]) if s is not None]
        identity[take.name] = float(np.mean(samples)) if samples else 0.0
        print(f"{take.name:8} similarity to character {identity[take.name]:+.3f}")

    assert dropout is not None
    hold_identity: list[float] = []
    hold_frames = 0
    hold_pasted = 0
    hold_flashes = 0
    if dropout.covered:
        first = dropout.times[dropout.covered.index(True)]
        for out, at, covered, pasted in zip(dropout.outputs, dropout.times,
                                            dropout.covered, dropout.pasted):
            if not covered or at - first > 0.40:
                continue
            hold_frames += 1
            if pasted is None:
                hold_flashes += 1  # engine gave up and the frame is the camera
                continue
            hold_pasted += 1
            value = identity_of(out, detector, embedder, character)
            if value is not None:
                hold_identity.append(value)
    held_mean = float(np.mean(hold_identity)) if hold_identity else 0.0
    print("\n--- dropout (face covered for 450 ms) ---")
    print(f"frames inside the 400 ms hold {hold_frames}, "
          f"pasted {hold_pasted}, character readable in {len(hold_identity)}, "
          f"similarity {held_mean:+.3f}, raw-camera frames {hold_flashes}")

    if args.save_frames:
        for take in takes:
            if take.outputs:
                cv2.imwrite(f"motion_{take.name}.png", take.outputs[len(take.outputs) // 2])
        if dropout.outputs:
            cv2.imwrite("motion_dropout.png", dropout.outputs[len(dropout.outputs) // 3])
        print("\nsaved motion_<take>.png")

    still = rows["still"]
    checks = {
        "still: paste sits on the detected face (<= 1.0 px)": still["glue"] <= 1.0,
        "still: swap does not swim (jitter <= 1.0 px)": still["jitter"] <= 1.0,
        "still: no drift (<= 0.20 px/s)": still["drift"] <= 0.20,
        "still: paste stays glued (p95 <= 2.0% eye distance)": still["glue_p95_pct"] <= 2.0,
        "still: paste stays glued (p95 <= 2.0% eye distance)": still["glue_p95_pct"] <= 2.0,
        # The paste may not be looser than the detector it is following: that is
        # what "glued to the performer" means in a measurement. The absolute
        # detector error against the scripted truth is separate (it is YuNet's
        # own noise on a resampled frame) and is reported, not asserted.
        "pan: paste follows detection (<= 1.15x)":
            rows["pan"]["glue"] <= 1.15 * max(1.0, rows["pan"]["det"]),
        "zoom: paste follows detection (<= 1.15x)":
            rows["zoom"]["glue"] <= 1.15 * max(1.0, rows["zoom"]["det"]),
        "roll: paste follows detection (<= 1.15x)":
            rows["roll"]["glue"] <= 1.15 * max(1.0, rows["roll"]["det"]),
        # 10% of the eye distance is roughly one frame of travel at the fastest
        # scripted pan (11.7 px/frame). Below that, per-frame glue is dominated
        # by YuNet's own scatter on a warped frame - the paste is a smoothed
        # path through noisy measurements and cannot be closer to each noisy
        # point without following the noise.
        "motion stays within 10% of the eye distance":
            max(rows[name]["glue_pct"] for name in MOTION_TAKES) <= 10.0,
        "pan: content travels with the head (rigidity <= 1.6)":
            rows["pan"]["rigid"] <= 1.6,
        "zoom: content travels with the head (rigidity <= 1.6)":
            rows["zoom"]["rigid"] <= 1.6,
        "roll: content travels with the head (rigidity <= 1.6)":
            rows["roll"]["rigid"] <= 1.6,
        "every motion frame was swapped":
            all(r["pasted"] == r["frames"] for r in rows.values()),
        "identity holds while moving (>= 0.50)": min(identity.values()) >= 0.50,
        "no take lags more than one frame": max(rows[n]["lag"] for n in MOTION_TAKES) <= 1,
        "dropout: engine keeps pasting (<= 1 frame given up)": hold_flashes <= 1,
        "dropout: character stays inside the hold": held_mean >= 0.50,
    }
    print()
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    ok = all(checks.values())
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
