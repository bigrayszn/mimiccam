# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Measure what a stale person matte costs at the hairline.

``MATTE_EVERY`` in the engine can halve the matte thread's work by queueing a
matte every second frame. The person's *colour* always comes from the current
frame, so what a stale matte changes is only the outline: where the subject has
moved into, the old silhouette leaves the room plate showing through them, and
where they have moved out of, the old silhouette leaves a strip of them behind.

This script puts a number on that. It builds a moving sequence from a still,
mattes every frame, then composites each frame twice:

    fresh   the matte computed from that frame            (MATTE_EVERY = 1)
    stale   the matte computed two frames earlier         (MATTE_EVERY = 2 at
                                                            30 fps means one
                                                            frame older than
                                                            this)

and reports how far apart they are, plus how much of the room shows through the
person in the stale version. It writes crops of the worst frame so the edge can
be looked at rather than argued about.

    python tests/verify_matte_lag.py
    python tests/verify_matte_lag.py --reference me.jpg --frame shot.jpg --fast
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mimiccam import face, matte, models, studio  # noqa: E402

# A difference this large in a video frame is the point where an edge starts to
# read as a doubled outline rather than as one soft edge.
VISIBLE_LEVELS = 8


def shifted(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate, keeping the frame full: what a lean towards the camera does."""
    height, width = frame.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (width, height), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def sequence(frame: np.ndarray, count: int, step: int):
    """A triangle-wave sway, so the subject stays framed like a real call."""
    span = 8
    for index in range(count):
        position = index % (2 * span)
        offset = position if position < span else 2 * span - position
        yield shifted(frame, offset * step, (offset * step) // 3)


def compare(scene: studio.StudioScene, frame: np.ndarray, box,
            fresh: tuple[np.ndarray, np.ndarray],
            stale: tuple[np.ndarray, np.ndarray]) -> dict:
    alpha_now, edge_now = fresh
    alpha_old, edge_old = stale
    out_now = scene.composite(frame, alpha_now, box, edge_now)
    out_old = scene.composite(frame, alpha_old, box, edge_old)

    difference = np.abs(out_now.astype(np.int16) - out_old.astype(np.int16)).max(axis=2)
    # The person moved into this area: the stale silhouette still shows the room
    # there, so this is the ghosting that a viewer would actually see.
    entered = (alpha_now > 0.5) & (alpha_old < 0.05)
    plate = scene.plate
    ghost_room = 0.0
    ghost_person = 0.0
    if entered.any():
        room = np.abs(out_old.astype(np.int16) - plate.astype(np.int16)).max(axis=2)
        honest = np.abs(out_now.astype(np.int16) - plate.astype(np.int16)).max(axis=2)
        ghost_room = float(room[entered].mean())
        ghost_person = float(honest[entered].mean())
    return {
        "mean": float(difference.mean()),
        "p99": float(np.percentile(difference, 99)),
        "max": float(difference.max()),
        "over_visible": int((difference > VISIBLE_LEVELS).sum()),
        "entered_px": int(entered.sum()),
        "ghost_room": ghost_room,
        "ghost_person": ghost_person,
        "out_now": out_now,
        "out_old": out_old,
        "difference": difference,
        "alpha_now": alpha_now,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=str(
        Path.home() / "AppData/Roaming/MimicCam/characters/"
        "WhatsApp Image 2026-09-15 at 1.17.18 PM.jpeg"))
    parser.add_argument("--frame", default=str(
        Path.home() / "AppData/Local/Temp/mimiccam-verify/biden.jpg"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fast", action="store_true",
                        help="brisk head turn (10 px a frame instead of 3)")
    args = parser.parse_args(argv)

    if not Path(args.reference).exists():
        raise SystemExit(f"reference image not found: {args.reference}")
    live = cv2.imread(args.frame)
    if live is None:
        raise SystemExit(f"could not read {args.frame}")
    base = cv2.resize(live, (1280, 720), interpolation=cv2.INTER_AREA)

    model = matte.MatteModel(models.path("matte"))
    plate, reference_face = studio.load_plate(args.reference, 1280, 720, model, None, print)
    scene = studio.StudioScene(plate, reference_face, print)

    # The composite needs a face box once to measure the seat, exactly as the
    # engine does on its first frame with a face.
    detector = face.FaceDetector(models.path("detector"), max_side=720)
    found = face.largest(detector.detect(base))
    if found is None:
        raise SystemExit(f"no face in {args.frame}")
    scene.seat_person(found.box)
    box = found.box

    step = 10 if args.fast else 3
    speed = step * 30
    print(f"\nmotion: {step} px a frame, about {speed} px/s at 30 fps\n")

    pairs: list[tuple[np.ndarray, tuple, tuple]] = []
    matte_ms: list[float] = []
    for frame in sequence(base, args.frames, step):
        started = time.perf_counter()
        result = model.matte(frame)
        matte_ms.append((time.perf_counter() - started) * 1000)
        pairs.append((frame, result, None))

    rows = []
    worst = None
    for index in range(2, len(pairs)):
        frame, fresh, _ = pairs[index]
        stale = pairs[index - 2][1]
        stats = compare(scene, frame, box, fresh, stale)
        rows.append(stats)
        if worst is None or stats["over_visible"] > worst["over_visible"]:
            worst = stats

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    print(f"frames compared           {len(rows)}")
    print(f"matte cost                {np.mean(matte_ms):.1f} ms per frame "
          f"(p95 {np.percentile(matte_ms, 95):.1f})")
    print("\n--- fresh matte vs a matte two frames old ---")
    print(f"mean difference           {mean('mean'):5.2f} levels")
    print(f"p99 difference            {mean('p99'):5.1f} levels")
    print(f"worst single pixel        {max(row['max'] for row in rows):5.0f} levels")
    print(f"pixels over {VISIBLE_LEVELS} levels     {int(np.mean([row['over_visible'] for row in rows]))} "
          f"of {base.shape[0] * base.shape[1]} a frame")
    entered = int(np.mean([row["entered_px"] for row in rows]))
    print(f"pixels the person entered {entered}")
    if entered:
        print(f"  room showing through    {mean('ghost_room'):5.1f} levels of difference "
              f"from the plate")
        print(f"  with a fresh matte      {mean('ghost_person'):5.1f} levels "
              f"(that is the person)")

    out_dir = ROOT
    crop_box = None
    if worst is not None:
        # Aim the crop at the pixel that disagrees most, in the output frame:
        # that is where an edge would double up.
        peak = int(np.argmax(worst["difference"]))
        top, left = (int(value) for value in np.unravel_index(peak, worst["difference"].shape))
        half = 90
        crop_box = (max(0, left - half), max(0, top - half),
                    min(base.shape[1], left + half), min(base.shape[0], top + half))
        x0, y0, x1, y1 = crop_box
        fresh_crop = worst["out_now"][y0:y1, x0:x1]
        stale_crop = worst["out_old"][y0:y1, x0:x1]
        zoom = 4
        fresh_zoom = cv2.resize(fresh_crop, None, fx=zoom, fy=zoom,
                                interpolation=cv2.INTER_NEAREST)
        stale_zoom = cv2.resize(stale_crop, None, fx=zoom, fy=zoom,
                                interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(out_dir / "matte_lag_fresh.png"), fresh_zoom)
        cv2.imwrite(str(out_dir / "matte_lag_stale.png"), stale_zoom)
        diff_crop = worst["difference"][y0:y1, x0:x1]
        cv2.imwrite(str(out_dir / "matte_lag_diff.png"),
                    np.clip(diff_crop * 12, 0, 255).astype(np.uint8))
        print(f"\nworst frame cropped at {crop_box} (4x, nearest): "
              f"{out_dir / 'matte_lag_fresh.png'} vs matte_lag_stale.png")

    # The engine cost of the two settings, measured the same way for both.
    from mimiccam.local_engine import LocalEngine  # noqa: PLC0415

    results = {}
    for every in (1, 2):
        engine = LocalEngine(reference_image=args.reference, width=1280, height=720,
                             fps=30, studio_background=True,
                             studio_image=args.reference, matte_every=every,
                             on_log=lambda _m: None)
        engine.start()
        if not engine.ready.wait(120):
            raise SystemExit("engine never became ready")
        import threading  # noqa: PLC0415

        rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        for _ in range(6):
            engine._process(rgb)
        stop = threading.Event()

        def producer() -> None:
            # 60 Hz so the engine is the limit, not the feed: at 30 Hz a quiet
            # machine simply keeps up with both settings and the test says
            # nothing.
            started = time.perf_counter()
            index = 0
            while not stop.is_set():
                target = started + index / 60.0
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
                engine.push(rgb, now)
                index += 1

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()
        time.sleep(4.0)
        stop.set()
        thread.join()
        results[every] = engine.measured_fps
        engine.stop()

    print("\n--- engine loop, full-frame subject at 720p ---")
    for every, fps in results.items():
        print(f"matte every {every} frame(s)      {fps:5.1f} fps")
    if results[1] > 0:
        print(f"gain from matting every other frame: "
              f"{results[2] - results[1]:+.1f} fps "
              f"({(results[2] / results[1] - 1) * 100:+.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
