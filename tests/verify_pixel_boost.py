# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Pixel boost vs the Detail pass, measured on one frame.

FaceFusion gets a larger face out of a 128 px network without a second model: it
warps the face to a bigger crop, cuts that crop into 128 px tiles, runs the swap
network on each tile and stitches the results back
(``facefusion/processors/pixel_boost.py``). This does the same with our own
``FaceSwapper`` and measures it against the GFPGAN pass that ships, so the
rejection recorded in FACE_ENHANCERS.md can be re-run rather than trusted.

    python tests/verify_pixel_boost.py
    python tests/verify_pixel_boost.py --character biden --target obama

What the numbers mean:

* ``detail`` is the Laplacian variance of the pasted face, measured in the frame
  at its own resolution, so the variants are compared on the same pixels.
* ``identity`` is ArcFace similarity between the pasted face and the character.
  This is the number that decides it: tiling raises detail and destroys identity,
  because each tile is a quadrant of a face and the network is being asked to
  swap something it cannot see in full.
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
from verify_local import fixture, webcam_framing  # noqa: E402


def tiled_render(swapper: face.FaceSwapper, frame: np.ndarray, landmarks: np.ndarray,
                 embedding: np.ndarray, tiles: int) -> np.ndarray:
    """Render at 128 * tiles px by running the network on each 128 px tile."""
    size = swapper.size * tiles
    matrix = swapper.matrix_for(landmarks, size)
    aligned = face.crop(frame, matrix, size)
    latent = swapper.latent(embedding)
    out = np.zeros((size, size, 3), np.uint8)
    step = swapper.size
    for ty in range(tiles):
        for tx in range(tiles):
            tile = aligned[ty * step:(ty + 1) * step, tx * step:(tx + 1) * step]
            blob = cv2.dnn.blobFromImage(
                tile, 1.0 / 255.0, (step, step), (0.0, 0.0, 0.0), swapRB=True
            )
            rendered = swapper.session.run(
                None, {swapper.input_names[0]: blob, swapper.input_names[1]: latent}
            )[0][0].transpose(1, 2, 0)
            out[ty * step:(ty + 1) * step, tx * step:(tx + 1) * step] = (
                np.clip(255.0 * rendered, 0, 255).astype(np.uint8)[:, :, ::-1]
            )
    # One tone match for the assembled crop: matching per tile made the seams
    # visible as four slightly different exposures.
    return face.match_tone(out, aligned)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--character", default="obama", choices=["obama", "biden"])
    parser.add_argument("--target", default="biden", choices=["obama", "biden"])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--strip", default="", help="write a side-by-side crop here")
    args = parser.parse_args()

    absent = models.missing()
    if absent:
        print(models.describe_missing(absent))
        return 2

    detector = face.FaceDetector(models.path("detector"))
    embedder = face.FaceEmbedder(models.path("identity"))
    swapper = face.FaceSwapper(models.path("swapper"), providers=["DmlExecutionProvider"],
                               keep_expression=False)
    enhancer = face.FaceEnhancer(models.path("enhancer"), providers=["DmlExecutionProvider"])

    reference = face.load_image(fixture("reference" if args.character == "obama" else "target"))
    ref_face = face.largest(detector.detect(reference))
    character = embedder.embedding(reference, ref_face.landmarks)

    frame = webcam_framing(face.load_image(fixture("target" if args.target == "biden" else "reference")),
                           detector)
    found = face.largest(detector.detect(frame))
    landmarks, box = found.landmarks, found.box

    variants = {
        "128 (shipped)": lambda: swapper.render(frame, landmarks, character),
        "128 + GFPGAN": lambda: enhancer.enhance(swapper.render(frame, landmarks, character)),
        "tiles 2x2 (256)": lambda: tiled_render(swapper, frame, landmarks, character, 2),
        "tiles 3x3 (384)": lambda: tiled_render(swapper, frame, landmarks, character, 3),
        "tiles 4x4 (512)": lambda: tiled_render(swapper, frame, landmarks, character, 4),
    }

    print(f"character {args.character}, target {args.target}, face box {box}")
    print(f"{'variant':16} {'render ms':>10} {'renders/s':>10} {'detail':>8} {'identity':>9}")
    shots: dict[str, np.ndarray] = {}
    for name, render in variants.items():
        render()  # warm up
        times = []
        texture = None
        for _ in range(max(1, args.repeat)):
            started = time.perf_counter()
            texture = render()
            times.append((time.perf_counter() - started) * 1000)
        pasted = swapper.paste(frame, texture, landmarks)
        x, y, w, h = box
        grey = cv2.cvtColor(pasted[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        detail = float(cv2.Laplacian(grey, cv2.CV_32F).var())
        pasted_face = face.largest(detector.detect(pasted))
        identity = (float(embedder.embedding(pasted, pasted_face.landmarks) @ character)
                    if pasted_face is not None else float("nan"))
        median = float(np.median(times))
        shots[name] = pasted
        print(f"{name:16} {median:10.1f} {1000.0 / median:10.1f} {detail:8.1f} "
              f"{identity:+9.3f}")

    if args.strip:
        x, y, w, h = box
        pad = int(0.25 * max(w, h))
        panels = []
        for name in ("128 (shipped)", "128 + GFPGAN", "tiles 2x2 (256)", "tiles 3x3 (384)"):
            crop = shots[name][max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
            panel = cv2.resize(crop, (300, 340), interpolation=cv2.INTER_AREA)
            cv2.putText(panel, name.replace(" (shipped)", "").replace(" (256)", ""),
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            panels.append(panel)
        strip = panels[0]
        for panel in panels[1:]:
            strip = np.hstack([strip, np.full((340, 6, 3), 255, np.uint8), panel])
        cv2.imwrite(args.strip, strip)
        print(f"wrote {args.strip}")

    print("\nRead the identity column: tiling raises detail and destroys the character.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
