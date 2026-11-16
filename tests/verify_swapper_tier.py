# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Before/after for the 256 px tier, at 100 % pixels and with numbers.

The claim under test is narrow: replacing inswapper's 128 px network with
HyperSwap's 256 px one, and putting XSeg in front of the paste, makes the same
photo look better on a call. So the harness holds everything else still - one
frame, one identity, one detector, one paste path - and varies only the two
things that changed.

Run it:

    python tests/verify_swapper_tier.py
    python tests/verify_swapper_tier.py --photo my.jpg --character face.jpg

It writes, into build/tmp:

``tier_compare_100.png``   the same face crop at 100 %, all four ways
``tier_occlusion.png``     a hand across the face, with and without XSeg
``tier_metrics.txt``       identity, sharpness and milliseconds per render
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mimiccam import face, models  # noqa: E402
from mimiccam.occlusion import Occluder  # noqa: E402

PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]
OUT = ROOT / "build" / "tmp"


def default_character() -> str:
    """The character photo the app is actually configured with, if any."""
    import json

    settings = Path(os.environ.get("APPDATA", "")) / "MimicCam" / "settings.json"
    try:
        chosen = json.loads(settings.read_text(encoding="utf-8")).get("reference_image", "")
    except (OSError, ValueError):
        chosen = ""
    if chosen and os.path.isfile(chosen):
        return chosen
    fallback = ROOT / "build" / "screenshots" / "reference.png"
    if fallback.is_file():
        return str(fallback)
    raise SystemExit("no character photo: pass --character path/to/face.jpg")


def _load(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"cannot read image: {path}")
    return image


def _label(canvas: np.ndarray, text: str, x: int, y: int,
           colour=(255, 255, 255), scale: float = 0.55) -> None:
    cv2.putText(canvas, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                colour, 1, cv2.LINE_AA)


def _crop(image: np.ndarray, box, pad: float = 0.30) -> np.ndarray:
    x, y, w, h = box
    side = int(max(w, h) * (1.0 + pad))
    cx, cy = x + w // 2, y + h // 2
    x0 = max(0, min(image.shape[1] - side, cx - side // 2))
    y0 = max(0, min(image.shape[0] - side, cy - side // 2))
    return image[y0:y0 + side, x0:x0 + side].copy()


def _score(image: np.ndarray, detector: face.FaceDetector,
           embedder: face.FaceEmbedder, character: np.ndarray) -> tuple[float, float]:
    """(identity cosine to the character, Laplacian sharpness) of one output."""
    found = face.largest(detector.detect(image))
    if found is None:
        return float("nan"), float("nan")
    got = embedder.embedding(image, found.landmarks)
    identity = float(np.dot(got, character))
    patch = _crop(image, found.box, pad=-0.25)
    grey = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return identity, float(cv2.Laplacian(grey, cv2.CV_64F).var())


def _timing(render, runs: int = 10) -> float:
    render()  # warm-up: DirectML compiles kernels on first call
    started = time.perf_counter()
    for _ in range(runs):
        render()
    return (time.perf_counter() - started) * 1000.0 / runs


def hand_over_face(image: np.ndarray, box, seed: int = 7) -> np.ndarray:
    """A fake hand and a hair strand, drawn over the face.

    Synthetic on purpose: the point of the test is that the mask responds to
    *something* in front of the face, and a drawn shape is repeatable from run to
    run in a way a photo of a real hand is not.
    """
    out = image.copy()
    x, y, w, h = box
    rng = np.random.default_rng(seed)
    # A hand: four fingers reaching up and left across the mouth and cheek from
    # the bottom right. Deliberately *over* the face, not beside it - a hand at
    # the edge of the crop is already outside the oval mask and would prove
    # nothing about XSeg.
    base = (int(x + w * 0.80), int(y + h * 0.99))
    for i, angle in enumerate((-118, -102, -86, -70)):
        length = int(h * (0.72 - 0.04 * i))
        tip = (int(base[0] + length * np.cos(np.deg2rad(angle))),
               int(base[1] + length * np.sin(np.deg2rad(angle))))
        # BGR, and deliberately a *skin* tone: the seam gate already refuses to
        # paste over pixels that differ wildly from the render, so a blue-grey
        # hand proves nothing about XSeg. A hand the same colour as the face is
        # what gets smeared, and is what the mask has to catch.
        cv2.line(out, base, tip, (118, 138, 168), int(w * 0.095), cv2.LINE_AA)
        cv2.circle(out, tip, int(w * 0.048), (122, 142, 172), -1, cv2.LINE_AA)
    cv2.circle(out, base, int(w * 0.125), (120, 140, 170), -1, cv2.LINE_AA)
    # A strand of hair across the forehead.
    pts = np.array([[x + w * 0.05, y + h * 0.02], [x + w * 0.55, y + h * 0.30],
                    [x + w * 1.02, y + h * 0.38]], np.int32)
    for j in range(3):
        shifted = pts + np.array([0, int(j * h * 0.02)])
        cv2.polylines(out, [shifted.reshape(-1, 1, 2)], False, (28, 22, 20), 7 + j,
                      cv2.LINE_AA)
    # Faint grain so the drawn shapes are not perfectly clean edges.
    noise = rng.normal(0, 3.0, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default=str(ROOT / "live_in_2.png"))
    parser.add_argument("--character", default="")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = _load(args.photo)
    character_path = args.character or default_character()
    character_image = _load(character_path)
    detector = face.FaceDetector(models.path("detector"), refine=False)
    embedder = face.FaceEmbedder(models.path("identity"), PROVIDERS)

    found = face.largest(detector.detect(target))
    if found is None:
        raise SystemExit("no face in the target photo")
    reference = face.largest(detector.detect(character_image))
    if reference is None:
        raise SystemExit("no face in the character photo")
    character = embedder.embedding(character_image, reference.landmarks)

    small = face.FaceSwapper(models.path("swapper"), PROVIDERS)
    large = face.FaceSwapper(models.swapper_path(), PROVIDERS)
    large.occluder = Occluder(models.path("occluder"), PROVIDERS)

    print(f"target    : {args.photo} {target.shape[1]}x{target.shape[0]}")
    print(f"character : {character_path}")
    print(f"tier      : {Path(models.swapper_path()).name} ({large.size} px)")

    plain = target
    before = small.swap(target, found.landmarks, character)
    after = large.swap(target, found.landmarks, character)

    # A clean 256 render with the occluder switched off, for the hand test.
    large_plain = face.FaceSwapper(models.swapper_path(), PROVIDERS)

    occluded = hand_over_face(target, found.box)
    occ_landmarks = found.landmarks
    occl_before = small.swap(occluded, occ_landmarks, character)
    occl_after_off = large_plain.swap(occluded, occ_landmarks, character)
    occl_after_on = large.swap(occluded, occ_landmarks, character)

    box = found.box
    # A clean face has nothing for XSeg to protect, so this sheet is the two
    # networks alone; the hand test below is where the mask earns its place.
    panels = [
        ("original", _crop(plain, box)),
        ("128 px inswapper (shipped)", _crop(before, box)),
        ("256 px HyperSwap", _crop(after, box)),
    ]
    height = max(p[1].shape[0] for p in panels)
    width = sum(p[1].shape[1] for p in panels) + 12 * (len(panels) + 1)
    sheet = np.full((height + 44, width, 3), 24, np.uint8)
    x = 12
    for name, patch in panels:
        sheet[32:32 + patch.shape[0], x:x + patch.shape[1]] = patch
        _label(sheet, name, x, 22)
        x += patch.shape[1] + 12
    cv2.imwrite(str(out_dir / "tier_compare_100.png"), sheet)

    hand_panels = [
        ("hand over face - 128 px", _crop(occl_before, box)),
        ("256 px, no XSeg", _crop(occl_after_off, box)),
        ("256 px + XSeg", _crop(occl_after_on, box)),
    ]
    height = max(p[1].shape[0] for p in hand_panels)
    width = sum(p[1].shape[1] for p in hand_panels) + 12 * (len(hand_panels) + 1)
    hand = np.full((height + 44, width, 3), 24, np.uint8)
    x = 12
    for name, patch in hand_panels:
        hand[32:32 + patch.shape[0], x:x + patch.shape[1]] = patch
        _label(hand, name, x, 22)
        x += patch.shape[1] + 12
    cv2.imwrite(str(out_dir / "tier_occlusion.png"), hand)

    rows = []
    for name, image in (("128 px inswapper", before), ("256 px HyperSwap", after),
                        ("256 + XSeg (hand)", occl_after_on),
                        ("256 no XSeg (hand)", occl_after_off),
                        ("128 (hand)", occl_before)):
        identity, sharpness = _score(image, detector, embedder, character)
        rows.append((name, identity, sharpness))

    ms_small = _timing(lambda: small.render(target, found.landmarks, character))
    ms_large = _timing(lambda: large.render(target, found.landmarks, character))
    ms_xseg = _timing(lambda: large.occluder.protection(
        face.crop(target, face.norm_matrix(found.landmarks, 256), 256)))

    lines = ["swap tier comparison", f"target: {args.photo}", ""]
    lines.append(f"{'variant':24s} {'identity':>9s} {'sharpness':>10s}")
    for name, identity, sharpness in rows:
        lines.append(f"{name:24s} {identity:9.3f} {sharpness:10.1f}")
    lines += [
        "",
        f"render 128 px : {ms_small:5.1f} ms",
        f"render 256 px : {ms_large:5.1f} ms",
        f"XSeg 256      : {ms_xseg:5.1f} ms (refreshed every {face.OCCLUSION_MAX_AGE * 1000:.0f} ms)",
        "",
        f"wrote {out_dir / 'tier_compare_100.png'}",
        f"wrote {out_dir / 'tier_occlusion.png'}",
    ]
    report = "\n".join(lines)
    (out_dir / "tier_metrics.txt").write_text(report, encoding="utf-8")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
