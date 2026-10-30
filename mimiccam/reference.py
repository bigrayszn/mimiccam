"""Conditioning a character photo for Lucy.

Decart's realtime documentation is specific about three things, and all three
change how well a character swap lands:

* pad unusual aspect ratios out to the output orientation instead of cropping
  the subject (the model reads the reference at the same shape as the video),
* keep the long side around 1280 px and never go under 512 px on the short one,
* describe the character in the prompt as well - a reference image sent alone
  is under-used, and the model fills the gap with whatever the person in the
  video is wearing.

The last point is why this module also reads the photo: the framing tells us
whether the reference can carry a full-body swap, and the sampled colours give
the prompt something concrete to name.
"""

from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MIN_SHORT_SIDE = 512
MAX_LONG_SIDE = 1280
OUTPUT_ASPECT = (16, 9)


@dataclass(frozen=True)
class Framing:
    """What the reference photo actually frames."""

    kind: str  # "full" | "half" | "portrait" | "unknown"
    face_fraction: float
    hint: str


@dataclass
class Appearance:
    """The parts of the character a prompt can name out loud.

    Hair and clothing are sampled; skin tone deliberately is not. A cheek band
    on a bearded face reads the beard as skin and named a tanned man "deep",
    and a wrong skin tone in the prompt is worse than no skin tone at all - the
    reference image already carries it.
    """

    hair: str = ""
    garments: list[str] = field(default_factory=list)
    framing: Framing = Framing("unknown", 0.0, "")

    def sentences(self) -> list[str]:
        out: list[str] = []
        if self.hair:
            out.append(f"They have {self.hair} hair.")
        if self.garments:
            joined = " and ".join(self.garments)
            out.append(f"They wear mostly {joined} clothing.")
        return out


# -- geometry --------------------------------------------------------------
def _open_rgb(path: str | Path):
    from PIL import Image

    with Image.open(path) as image:
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            backdrop = Image.new("RGBA", image.size, (32, 34, 38, 255))
            backdrop.alpha_composite(image)
            image = backdrop
        return image.convert("RGB")


def _pad_to_aspect(image, ratio: float):
    """Pad, never crop: the subject must keep every pixel it arrived with.

    The padding is a heavy blur of the photo itself rather than a flat colour.
    A hard bar of grey reads to the model as a frame, and it shows up in the
    output as a letterbox; a soft extension reads as out-of-focus room.
    """
    from PIL import Image, ImageFilter

    width, height = image.size
    if abs(width / height - ratio) < 0.01:
        return image
    if width / height < ratio:
        target = (max(width, int(round(height * ratio))), height)
    else:
        target = (width, max(height, int(round(width / ratio))))
    backdrop = image.resize(target, Image.LANCZOS).filter(
        ImageFilter.GaussianBlur(radius=max(4.0, target[1] / 40.0))
    )
    backdrop.paste(image, ((target[0] - width) // 2, (target[1] - height) // 2))
    return backdrop


def prepare_reference(path: str | Path, aspect: tuple[int, int] = OUTPUT_ASPECT) -> bytes:
    """Character photo -> JPEG bytes sized and shaped the way Lucy wants.

    Raises ``FileNotFoundError`` for a missing file: a missing reference must be
    a clear failure, not a silent prompt with no character in it.
    """
    from PIL import Image

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(path))
    image = _open_rgb(source)
    image = _pad_to_aspect(image, aspect[0] / aspect[1])

    width, height = image.size
    short_side = min(width, height)
    if short_side < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / short_side
        image = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.LANCZOS,
        )
    if max(image.size) > MAX_LONG_SIDE:
        image.thumbnail((MAX_LONG_SIDE, MAX_LONG_SIDE), Image.LANCZOS)

    # A light unsharp pass recovers the detail that the resize above softened;
    # Decart asks for a sharp reference and sharpness is most of identity.
    array = np.asarray(image, dtype=np.uint8)
    blurred = cv2.GaussianBlur(array, (0, 0), 1.1)
    array = cv2.addWeighted(array, 1.35, blurred, -0.35, 0)

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=93)
    return buffer.getvalue()


# -- reading the photo -----------------------------------------------------
def _face_box(bgr: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """Largest face as x, y, w, h. None when the detector is unavailable."""
    from . import models

    model = models.path("detector")
    if not model.exists():
        return None
    height, width = bgr.shape[:2]
    try:
        detector = cv2.FaceDetectorYN.create(str(model), "", (width, height), 0.6, 0.3, 50)
        _count, faces = detector.detect(bgr)
    except (cv2.error, AttributeError):
        return None
    if faces is None or len(faces) == 0:
        return None
    best = max(faces, key=lambda row: row[2] * row[3])
    x, y, w, h = (int(round(v)) for v in best[:4])
    return max(0, x), max(0, y), max(1, w), max(1, h)


def _skin_mask(rgb: np.ndarray) -> np.ndarray:
    """YCrCb skin bounds, the same idea the body/hair pass uses."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return (mask.astype(np.uint8) * 255)


def _name(rgb: tuple[int, int, int]) -> str:
    r, g, b = (channel / 255.0 for channel in rgb)
    hue, sat, val = colorsys.rgb_to_hsv(r, g, b)
    if val < 0.16:
        return "black"
    if sat < 0.12:
        if val > 0.82:
            return "white"
        return "light grey" if val > 0.55 else "grey"
    if val < 0.30:
        return "black"
    degrees = hue * 360.0
    if degrees < 15 or degrees >= 345:
        base = "red"
    elif degrees < 40:
        base = "orange"
    elif degrees < 68:
        base = "yellow"
    elif degrees < 165:
        base = "green"
    elif degrees < 200:
        base = "teal"
    elif degrees < 250:
        base = "blue"
    elif degrees < 290:
        base = "purple"
    else:
        base = "pink"
    if sat < 0.30:
        return "grey" if val < 0.6 else "pale " + base
    if val < 0.50:
        return "dark " + base
    return base


def _median_colour(rgb: np.ndarray, mask: np.ndarray) -> Optional[tuple[int, int, int]]:
    pixels = rgb[mask > 0]
    if len(pixels) < 32:
        return None
    median = np.median(pixels, axis=0)
    return int(median[0]), int(median[1]), int(median[2])


def _dominant_colours(rgb: np.ndarray, mask: np.ndarray, k: int = 3) -> list[str]:
    """Name the colours that actually cover the clothing region.

    Named counts, not k-means: the clusters split one garment into two shades
    and then a shirt covering 11% of the torso fell under the area threshold and
    vanished from the prompt.
    """
    pixels = rgb[mask > 0]
    if len(pixels) < 64:
        return []
    sample = pixels[:: max(1, len(pixels) // 4000)]
    counts: dict[str, int] = {}
    for pixel in sample:
        name = _name((int(pixel[0]), int(pixel[1]), int(pixel[2])))
        counts[name] = counts.get(name, 0) + 1
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return [name for name, count in ranked if count / total >= 0.08][:k]


def _framing(face: Optional[tuple[int, int, int, int]], height: int) -> Framing:
    if face is None:
        return Framing(
            "unknown",
            0.0,
            "Could not find a face in the character photo - Lucy will lean on the "
            "prompt alone. A clear, evenly lit photo of the face works best.",
        )
    fraction = face[3] / max(1, height)
    if fraction >= 0.30:
        return Framing(
            "portrait",
            fraction,
            "Chest-up reference: Lucy knows the character's face and jacket, not "
            "their trousers or shoes. Name the lower half in the look notes (for "
            "example \"black trousers, black leather shoes\") or use a full-body "
            "photo - otherwise the model keeps your own lower body.",
        )
    if fraction >= 0.14:
        return Framing(
            "half",
            fraction,
            "Waist-up reference: the torso is covered. Add the trousers and shoes "
            "to the look notes so the legs change too.",
        )
    return Framing(
        "full",
        fraction,
        "Full-body reference: the character's build, outfit and footwear all come "
        "from the photo.",
    )


def analyse(path: str | Path) -> Appearance:
    """Read framing, hair, skin and clothing colours off a character photo."""
    appearance = Appearance()
    try:
        rgb = np.asarray(_open_rgb(path), dtype=np.uint8)
    except (OSError, ValueError):
        return appearance
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    face = _face_box(bgr)
    appearance.framing = _framing(face, rgb.shape[0])
    if face is None:
        return appearance

    fx, fy, fw, fh = face
    height, width = rgb.shape[:2]

    # Hair: a band hugging the top of the head box, where hair has to be for the
    # face to be visible. The median survives a bright wall behind it.
    top = max(0, fy - int(0.35 * fh))
    band = np.zeros((height, width), dtype=np.uint8)
    band[top:max(top + 1, fy + int(0.10 * fh)),
         max(0, fx + int(0.10 * fw)):min(width, fx + int(0.90 * fw))] = 255
    skin = _skin_mask(rgb)
    band[skin > 0] = 0
    colour = _median_colour(rgb, band)
    if colour is not None:
        appearance.hair = _name(colour)

    # Clothing: everything below the chin, excluding skin (neck, chest, arms).
    torso = np.zeros((height, width), dtype=np.uint8)
    torso[min(height - 1, fy + int(1.02 * fh)):height,
          max(0, fx - int(0.60 * fw)):min(width, fx + int(1.60 * fw))] = 255
    torso[skin > 0] = 0
    appearance.garments = _dominant_colours(rgb, torso)
    return appearance
