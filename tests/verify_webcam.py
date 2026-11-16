# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""How the swap reads at the size a call actually shows it.

The engine is tuned and measured at 1280x720, but nobody watches it at that
size: a one-on-one call shows the other person in a tile around 640x360, and a
group call drops to 320x180 or less. Two things behave differently at that size.

* Detail stops mattering. GFPGAN's texture is only a few pixels wide in a tile,
  so it averages away; what survives the downscale is the low-frequency picture
  - the shading, the tone of the face against the neck, and the outline of the
  mask.
* Anything that moves between frames survives perfectly. A face whose brightness
  steps a couple of levels when the head moves is invisible in a 720p still and
  reads as a flicker in a small tile, which is exactly what makes a viewer look
  twice.

So this runs the real engine over a moving frame and measures, at each size:

    identity   ArcFace similarity to the character photo, from the tile itself
    flicker    frame-to-frame change in the face box, and in an 8x-downscaled
               copy of it (the part of the change the eye sees at tile size)
    luma       the face's own brightness over the run, and how far it drifts
    seam       gradient energy in the paste's soft edge against the face
               interior; near 1.0 means the mask outline is not a visible line

The same measurements are taken on the *input* tiles, so every number has a
floor: a still frame panned by a few pixels is what the engine starts from, and
anything above it is the engine's own doing.

    python tests/verify_webcam.py --source live.png --reference photo.jpg
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
sys.path.insert(0, str(ROOT / "tests"))

from mimiccam import face, models  # noqa: E402
from mimiccam.local_engine import LocalEngine  # noqa: E402
from verify_local import animate  # noqa: E402

DEFAULT_REFERENCE = Path.home() / "AppData" / "Roaming" / "MimicCam" / "characters" / "flowz2.jpeg"
TILES = ((1280, 720), (960, 540), (640, 360), (480, 270), (320, 180), (240, 135))


def ramp_mask(shape: tuple[int, int], swapper, landmarks: np.ndarray,
              size: int) -> np.ndarray:
    """The paste's soft edge, in frame pixels, as uint8 alpha.

    Rebuilt from the same matrix the swapper pastes with, so the band this
    measures is the band the composite actually blended.
    """
    inverse = cv2.invertAffineTransform(swapper.matrix_for(landmarks, size))
    mask = np.clip(swapper.mask_for(size) * 255.0, 0, 255).astype(np.uint8)
    return cv2.warpAffine(mask, inverse, (shape[1], shape[0]), flags=cv2.INTER_LINEAR)


def seam_ratio(tile_bgr: np.ndarray, alpha: np.ndarray) -> float:
    """Gradient energy in the mask ramp against the face interior.

    A seam is a line: it shows up as edge energy where the blend is half
    transparent, on a face that is otherwise smooth shading. Normalising by the
    face's own texture energy keeps the number comparable between a bearded and
    a bare face - what it asks is whether the outline stands out *from the
    face*, not whether the face has detail.
    """
    grey = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    band = cv2.inRange(alpha, 10, 245)
    inside = cv2.erode((alpha > 245).astype(np.uint8), np.ones((5, 5), np.uint8))
    if cv2.countNonZero(inside) < 200 or cv2.countNonZero(band) < 100:
        return float("nan")
    edge = float(cv2.mean(gradient, mask=band)[0])
    texture = float(np.median(gradient[inside > 0]))
    if texture < 1e-3:
        return float("nan")
    return edge / texture


def detail_ratio(tile_bgr: np.ndarray, points: np.ndarray) -> float:
    """High-frequency energy inside the face against the neck below it.

    The eye does not compare a face with an abstract notion of sharpness, it
    compares it with the next piece of skin it can see. A ratio near 1 means the
    pasted face and the untouched neck carry the same amount of camera detail; a
    face that reads as plastic is one where this sits well below it.
    """
    height, width = tile_bgr.shape[:2]
    span = float(np.linalg.norm(points[1] - points[0]))
    if span < 6.0:
        return float("nan")
    centre = points[2]
    radius = max(3, int(span * 0.55))
    neck_x = int((points[3][0] + points[4][0]) / 2)
    neck_y = int((points[3][1] + points[4][1]) / 2 + span * 0.75)
    grey = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(grey, cv2.CV_32F, ksize=3))
    face = np.zeros(grey.shape, dtype=np.uint8)
    cv2.circle(face, (int(centre[0]), int(centre[1] - radius * 0.2)), radius, 255, -1)
    half = max(2, int(span * 0.35))
    x0, x1 = max(0, neck_x - half), min(width, neck_x + half)
    y0, y1 = max(0, neck_y - half), min(height, neck_y + half)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return float("nan")
    neck = lap[y0:y1, x0:x1]
    face_value = float(np.median(lap[face > 0]))
    neck_value = float(np.median(neck))
    if neck_value < 1e-3:
        return float("nan")
    return face_value / neck_value


def measure(frames: list[np.ndarray], tiles_source: list[np.ndarray],
            character: np.ndarray, detector: face.FaceDetector,
            embedder: face.FaceEmbedder, swapper, tile: tuple[int, int],
            out_dir: Path, full_landmarks: list[np.ndarray]) -> dict:
    """Every number, at one size, against the same measurement on the input.

    Identity is aligned from the *full-size* detection, scaled into the tile,
    rather than re-detected at the small size: a detector that lands three
    pixels differently between two tile sizes moves ArcFace's crop enough to
    swing the score by 0.15, which would be measured as the tile size mattering
    when it is the detector. The eye does not re-detect; it sees the same face
    with fewer pixels.
    """
    width, height = tile
    scale = width / float(frames[0].shape[1])
    tiles = [cv2.resize(frame, tile, interpolation=cv2.INTER_AREA) for frame in frames]
    source_tiles = [cv2.resize(frame, tile, interpolation=cv2.INTER_AREA)
                    for frame in tiles_source]
    small = [cv2.resize(tile_bgr, (max(1, width // 8), max(1, height // 8)),
                        interpolation=cv2.INTER_AREA) for tile_bgr in tiles]
    source_small = [cv2.resize(tile_bgr, (max(1, width // 8), max(1, height // 8)),
                               interpolation=cv2.INTER_AREA) for tile_bgr in source_tiles]

    if not full_landmarks:
        return {"tile": f"{width}x{height}", "faces": 0}
    points = [landmarks * scale for landmarks in full_landmarks]
    box = (int(points[-1][:, 0].min()), int(points[-1][:, 1].min()),
           int(np.ptp(points[-1][:, 0])), int(np.ptp(points[-1][:, 1])))
    x, y, w, h = box
    # The face, generously: an oval around the five points, not the jaw line.
    x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
    x1, y1 = min(width, x + w + w // 2), min(height, y + h + h // 2)
    region = (slice(y0, y1), slice(x0, x1))

    # Every eighth frame: an ArcFace pass is ~10 ms on the CPU and a train of
    # them only makes the harness slower without changing what it reports.
    step = max(1, len(tiles) // 10)
    identities = [float(embedder.embedding(tile_bgr, landmark) @ character)
                  for tile_bgr, landmark in list(zip(tiles, points))[::step]]
    source_identities = [float(embedder.embedding(tile_bgr, landmark) @ character)
                         for tile_bgr, landmark in list(zip(source_tiles, points))[::step]]

    steps = [float(np.abs(a[region].astype(np.int16) - b[region].astype(np.int16)).mean())
             for a, b in zip(tiles, tiles[1:])]
    source_steps = [
        float(np.abs(a[region].astype(np.int16) - b[region].astype(np.int16)).mean())
        for a, b in zip(source_tiles, source_tiles[1:])
    ]
    low_steps = [float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
                 for a, b in zip(small, small[1:])]
    source_low_steps = [
        float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
        for a, b in zip(source_small, source_small[1:])
    ]

    lumas = np.array([float(cv2.cvtColor(tile_bgr[region], cv2.COLOR_BGR2GRAY).mean())
                      for tile_bgr in tiles if tile_bgr[region].size])
    source_lumas = np.array([
        float(cv2.cvtColor(tile_bgr[region], cv2.COLOR_BGR2GRAY).mean())
        for tile_bgr in source_tiles if tile_bgr[region].size
    ])

    middle = len(frames) // 2
    detail = detail_ratio(tiles[middle], points[middle])
    source_detail = detail_ratio(source_tiles[middle], points[middle])
    seam, source_seam = float("nan"), float("nan")
    full = face.largest(detector.detect(frames[middle]))
    if full is not None:
        modelled = int(round(w * 128.0 / max(1.0, float(full.box[2]))))
        for size in (modelled, 128, 512):
            if size <= 0:
                continue
            alpha = ramp_mask(frames[middle].shape[:2], swapper, full.landmarks, size)
            alpha_small = cv2.resize(alpha, tile, interpolation=cv2.INTER_AREA)
            seam = seam_ratio(tiles[middle], alpha_small)
            source_seam = seam_ratio(source_tiles[middle], alpha_small)
            if not np.isnan(seam):
                break

    cv2.imwrite(str(out_dir / f"tile_{width}x{height}.png"), tiles[middle])
    cv2.imwrite(str(out_dir / f"source_tile_{width}x{height}.png"), source_tiles[middle])
    return {
        "tile": f"{width}x{height}",
        "identity": float(np.mean(identities)),
        "source_identity": float(np.mean(source_identities)),
        "identity_min": float(np.min(identities)),
        "flicker": float(np.mean(steps)),
        "flicker_max": float(np.max(steps)),
        "source_flicker": float(np.mean(source_steps)),
        "low_flicker": float(np.mean(low_steps)),
        "source_low": float(np.mean(source_low_steps)),
        "luma": float(np.mean(lumas)),
        "luma_span": float(np.max(lumas) - np.min(lumas)),
        "luma_sd": float(np.std(lumas)),
        "source_span": float(np.max(source_lumas) - np.min(source_lumas)),
        "source_sd": float(np.std(source_lumas)),
        "seam": seam,
        "source_seam": source_seam,
        "detail": detail,
        "source_detail": source_detail,
        "faces": len(frames),
        "frames": len(tiles),
        "box": box,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT / "build" / "tmp" / "look" / "live.png"),
                        help="a webcam frame, resized to 1280x720")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE),
                        help="the character photo")
    parser.add_argument("--seconds", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=40,
                        help="frames to drop before measuring (texture settle)")
    parser.add_argument("--out", default=str(ROOT / "build" / "tmp" / "webcam"))
    parser.add_argument("--detail", choices=("on", "off"), default="on")
    parser.add_argument("--legacy", action="store_true",
                        help="measure what the engine did before the paste's colour "
                             "was gated to skin and locked across frames and its "
                             "texture folded, for a before/after of the same take")
    parser.add_argument("--still", action="store_true",
                        help="hold the head still instead of panning it: what a call "
                             "looks like most of the time, and where texture speckle "
                             "shows")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = cv2.imread(args.source)
    if base is None:
        raise SystemExit(f"could not read {args.source}")
    base = cv2.resize(base, (1280, 720), interpolation=cv2.INTER_AREA)
    count = args.warmup + int(args.seconds * 30)
    frames = [base.copy() for _ in range(count)] if args.still else animate(base, count)

    logs: list[str] = []
    engine = LocalEngine(
        reference_image=args.reference,
        preset_key="character",
        width=1280,
        height=720,
        on_log=logs.append,
        body_swap=True,
        keep_expression=True,
        detail=args.detail == "on",
    )
    started = time.perf_counter()
    engine.start()
    if args.legacy and engine._swapper is not None:
        # The old paste: the whole ramp, no gate, and every frame's reading applied
        # as it was measured. Only ever used to produce a before/after of one take.
        face.SEAM_GATE = (0.0, 99.0)
        engine._swapper._tone_lock = face.ToneLock(rate=1.0, slew=255.0)
    print(f"engine ready in {time.perf_counter() - started:.1f}s "
          f"(provider {engine.provider}, swapping {engine.swapping})")

    grabbed: list[np.ndarray] = []
    used: list[np.ndarray] = []
    last_out = 0.0
    for index, frame in enumerate(frames):
        engine.push(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), time.perf_counter())
        time.sleep(1 / 30)
        if index >= args.warmup:
            got, got_ts = engine.latest()
            # Timestamp, not pixel equality: a still head can legitimately
            # produce byte-identical output, and that is a result, not a repeat.
            if got is not None and got_ts != last_out:
                last_out = got_ts
                grabbed.append(cv2.cvtColor(got, cv2.COLOR_RGB2BGR))
                used.append(frame)
    engine.stop()
    print(f"captured {len(grabbed)} frames, engine {engine.measured_fps:.1f} fps, "
          f"render {engine.render_fps:.1f} fps, latency {engine.latency_ms():.0f} ms")

    detector = face.FaceDetector(models.path("detector"), max_side=720, refine=True)
    embedder = face.FaceEmbedder(models.path("identity"), ["CPUExecutionProvider"])
    photo = face.load_image(args.reference)
    photo_face = face.largest(detector.detect(photo))
    character = embedder.embedding(photo, photo_face.landmarks)
    original = face.largest(detector.detect(base))
    original_vec = (embedder.embedding(base, original.landmarks)
                    if original is not None else np.zeros(512, np.float32))
    print(f"character vs the person on camera: {float(character @ original_vec):.3f}")

    # A swapper only for its mask geometry; the engine already closed its own.
    swapper = face.FaceSwapper(models.path("swapper"), ["CPUExecutionProvider"])
    kept = []
    for out_frame, source_frame in zip(grabbed, used):
        found = face.largest(detector.detect(out_frame))
        if found is not None:
            kept.append((out_frame, source_frame, found.landmarks))
    print(f"face found in {len(kept)}/{len(grabbed)} full-size frames")
    kept_out = [item[0] for item in kept]
    kept_source = [item[1] for item in kept]
    full_landmarks = [item[2] for item in kept]

    rows = [measure(kept_out, kept_source, character, detector, embedder, swapper, tile,
                    out_dir, full_landmarks) for tile in TILES]
    print(f"{'tile':>10} {'ident':>6} {'min':>6} {'src':>6} {'flick':>6} {'src':>6} "
          f"{'max':>5} {'low':>5} {'src':>5} {'luma':>5} {'sd':>5} {'src':>5} "
          f"{'seam':>5} {'src':>5} {'tex':>5} {'src':>5}")
    for row in rows:
        if not row.get("faces"):
            print(f"{row['tile']:>10}  no face")
            continue
        print(f"{row['tile']:>10} {row['identity']:>6.3f} {row['identity_min']:>6.3f} "
              f"{row['source_identity']:>6.3f} {row['flicker']:>6.2f} "
              f"{row['source_flicker']:>6.2f} {row['flicker_max']:>5.2f} "
              f"{row['low_flicker']:>5.2f} {row['source_low']:>5.2f} "
              f"{row['luma']:>5.1f} {row['luma_sd']:>5.2f} {row['source_sd']:>5.2f} "
              f"{row['seam']:>5.2f} {row['source_seam']:>5.2f} "
              f"{row['detail']:>5.2f} {row['source_detail']:>5.2f}")
    for line in logs:
        print("log|", line)

    # A 1:1 strip of the call-sized tiles, so the numbers can be looked at.
    strips = []
    for width, height in TILES[2:]:
        image = cv2.imread(str(out_dir / f"tile_{width}x{height}.png"))
        if image is not None:
            strips.append(cv2.copyMakeBorder(image, 0, 0, 0, 8, cv2.BORDER_CONSTANT,
                                             value=(40, 40, 40)))
    if strips:
        tallest = max(item.shape[0] for item in strips)
        padded = [cv2.copyMakeBorder(item, 0, tallest - item.shape[0], 0, 0,
                                     cv2.BORDER_CONSTANT, value=(40, 40, 40))
                  for item in strips]
        cv2.imwrite(str(out_dir / "tiles.png"), np.hstack(padded))
        print(f"wrote {out_dir / 'tiles.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
