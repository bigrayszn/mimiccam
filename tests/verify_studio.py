# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""End-to-end check of the studio background.

Runs the real models on a real frame and reports what changed:

* the room plate build (first time) and its cache hit
* swapped identity before and after the composite
* matte and composite cost per frame, and the resulting ceiling FPS
* how much of the camera's own background survives outside the person

    python tests/verify_studio.py --frame shot.jpg
    python tests/verify_studio.py --camera 1
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

from mimiccam import body, face, matte, models, studio  # noqa: E402


def grab(camera: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera, 700)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {camera}")
    frame = None
    deadline = time.time() + 8
    while time.time() < deadline:
        ok, candidate = cap.read()
        if ok and candidate is not None and candidate.mean() > 20:
            frame = candidate
            break
        time.sleep(0.1)
    cap.release()
    if frame is None:
        raise SystemExit("Camera delivered no lit frame")
    return frame


def identity(frame: np.ndarray, detector, embedder, target: np.ndarray):
    found = face.largest(detector.detect(frame))
    if found is None:
        return None
    return float(np.dot(embedder.embedding(frame, found.landmarks), target))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default="", help="still to use as the camera")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--reference", default="")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args(argv)

    reference = args.reference or str(
        Path.home() / "AppData/Roaming/MimicCam/characters/"
        "WhatsApp Image 2026-09-15 at 1.17.18 PM.jpeg"
    )
    if not Path(reference).exists():
        raise SystemExit(f"reference image not found: {reference}")

    with open(reference, "rb"):
        pass

    print("models:", ", ".join(spec.filename for spec in models.SPECS))
    detector = face.FaceDetector(models.path("detector"), max_side=720)
    embedder = face.FaceEmbedder(models.path("identity"))
    swapper = face.FaceSwapper(models.path("swapper"))
    print("swap provider:", swapper.session.get_providers()[0])

    reference_image = face.load_image(reference)
    target_embedding = face.identity_from_photo(reference_image, detector, embedder)
    target_face = face.largest(detector.detect(reference_image))

    live = cv2.imread(args.frame) if args.frame else grab(args.camera)
    if live is None:
        raise SystemExit(f"could not read {args.frame}")
    frame = cv2.resize(live, (1280, 720), interpolation=cv2.INTER_AREA)
    found = face.largest(detector.detect(frame))
    if found is None:
        raise SystemExit("no face in the camera frame")
    before = identity(frame, detector, embedder, target_embedding)

    started = time.perf_counter()
    matte_model = matte.MatteModel(models.path("matte"))
    print(f"matte session            {time.perf_counter() - started:.1f} s "
          f"({matte_model.provider})")
    plate_started = time.perf_counter()
    cached = studio.is_plate_cached(reference, 1280, 720)
    inpaint_model = None if cached else studio.InpaintModel(models.path("inpaint"))
    plate, reference_face = studio.load_plate(
        reference, 1280, 720, matte_model, inpaint_model, print
    )
    print(f"plate                    {time.perf_counter() - plate_started:.1f} s "
          f"({'cache' if cached else 'built'}), reference face {reference_face}")
    scene = studio.StudioScene(plate, reference_face, print)

    # -- swap one face, then seat it ------------------------------------
    texture = swapper.render(frame, found.landmarks, target_embedding)
    swapped = swapper.paste(frame, texture, found.landmarks)
    tone = body.face_skin_tone(swapped, found.box)
    if tone is not None:
        swapped = body.harmonize(swapped, found.box, tone)
    after_swap = identity(swapped, detector, embedder, target_embedding)

    alpha, edge = matte_model.matte(swapped)
    started = time.perf_counter()
    out = scene.composite(swapped, alpha, found.box, edge)
    first_ms = (time.perf_counter() - started) * 1000

    cv2.imwrite(str(ROOT / "studio_frame.png"), out)
    cv2.imwrite(str(ROOT / "studio_source.png"), swapped)

    # -- per-frame budget -------------------------------------------------
    timings = {"matte": [], "composite": []}
    for index in range(args.frames):
        shifted = np.roll(swapped, index % 5 - 2, axis=1)
        started = time.perf_counter()
        a, e = matte_model.matte(shifted)
        timings["matte"].append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        scene.composite(shifted, a, found.box, e)
        timings["composite"].append((time.perf_counter() - started) * 1000)

    per_frame = (np.mean(timings["matte"]) + np.mean(timings["composite"]))
    print("\n--- cost ---")
    print(f"matte                    {np.mean(timings['matte']):.1f} ms "
          f"(p95 {np.percentile(timings['matte'], 95):.1f})")
    print(f"composite                {np.mean(timings['composite']):.1f} ms "
          f"(first {first_ms:.1f}, p95 {np.percentile(timings['composite'], 95):.1f})")
    print(f"studio ceiling           {1000.0 / per_frame:.1f} fps before the swap pass")

    after = identity(out, detector, embedder, target_embedding)

    # -- did any of the camera room survive? ------------------------------
    # Measured on the warped matte, not the camera-space one: the seat scales
    # and shifts the person, so a camera pixel that is background can sit where
    # the pasted person is opaque. That made this number move with the person's
    # sharpness instead of with the background.
    _, seated_alpha, origin = scene._warp(swapped, alpha, scene.seat.matrix())
    outside = np.zeros((out.shape[0], out.shape[1]), dtype=bool)
    origin_x, origin_y = origin
    height, width = seated_alpha.shape[:2]
    region = outside[origin_y:origin_y + height, origin_x:origin_x + width]
    region[:] = seated_alpha[:region.shape[0], :region.shape[1]] < 0.02
    print(f"\noutside-person area {outside.mean() * 100:.0f}% of the final frame")
    camera_side = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    output_side = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    plate_side = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    frame_sim = float(np.corrcoef(camera_side[outside], output_side[outside])[0, 1])
    plate_sim = float(np.corrcoef(plate_side[outside], output_side[outside])[0, 1])
    # The plate and the camera frame are two photographs of similar rooms, so
    # they correlate a little on their own. Comparing the output against that
    # baseline is what says whether any camera pixels survived: output ~= plate
    # means the number below is the baseline, not leakage.
    baseline = float(np.corrcoef(camera_side[outside], plate_side[outside])[0, 1])

    print("\n--- identity ---")
    print(f"camera frame             {before:+.3f}")
    print(f"after face swap          {after_swap:+.3f}")
    print(f"after studio composite   {after if after is not None else float('nan'):+.3f}")
    print("\n--- background ---")
    print(f"outside-person pixels vs camera frame  {frame_sim:+.3f} "
          f"(plate alone scores {baseline:+.3f}, so ~0 is the floor)")
    print(f"outside-person pixels vs room plate    {plate_sim:+.3f} (want ~+1)")
    print(f"surviving camera background           "
          f"{'none' if plate_sim > 0.995 else 'present - check the matte'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
