"""Studio background: the reference photo's room, with you in it.

The reference image is a single still: a person seated in a room. Using it as a
live background needs three separate jobs, and they are easy to confuse:

1. **Remove the reference person.** Their pixels are not ours to leave behind.
   An image matte finds them, then LaMa (a large-mask inpainting network) paints
   the room back over the hole in one 512x512 pass. The result is cached: the
   pass costs a few seconds of CPU the first time (measured 6.9 s on a loaded
   i9) and nothing afterwards.

2. **Seat the live person.** The camera person is scaled and anchored at the
   bottom of the frame so they occupy the chair the reference person used. The
   transform is measured once from the face and then *held*, because a body that
   rescales every time the head moves looks like it is sliding on the chair.

3. **Make the two belong together.** The person is relit towards the room's
   ambient colour, their edge is unmixed from the camera background (see
   :mod:`mimiccam.matte`), and a soft shadow is cast onto the plate behind them.

Everything here is a still-image reconstruction plus a per-frame composite. No
generative model is asked to redraw the person, so head turns, blinks, lip
movement and gestures survive untouched - which is the point.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from . import face, matte

PLATE_VERSION = "plate-v3"
INPAINT_SIZE = (512, 512)
# Match the reference face width: the seat is what the reference photo defines,
# so a different face size is a different distance from the camera.
SEAT_GAIN = 1.0
MAX_SCALE = 1.55
MIN_SCALE = 0.55
SHADOW_STRENGTH = 0.38


class StudioError(RuntimeError):
    """The studio background cannot be built (bad image, missing model)."""


# -- inpainting ------------------------------------------------------------
_INPAINT: dict[str, "InpaintModel"] = {}


def shared_inpaint(model_path: str | Path) -> "InpaintModel":
    """One LaMa session per process: it takes seconds to build and is reusable."""
    key = str(model_path)
    model = _INPAINT.get(key)
    if model is None:
        model = InpaintModel(key)
        _INPAINT[key] = model
    return model


class InpaintModel:
    """LaMa, used once per reference image to erase the reference person.

    DirectML rejects this graph's Fourier MatMul (the provider returns
    ``80070057``), so it deliberately runs on the CPU: one pass of a few seconds
    that is then cached is cheaper than another 200 MB model that works on the
    GPU.
    """

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 4
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {item.name for item in self.session.get_inputs()}
        if not {"image", "mask"} <= self._inputs:
            raise StudioError("inpainting model does not expose image/mask inputs")

    def fill(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return the image with everything under ``mask`` repainted."""
        if not mask.any():
            return image_bgr
        h, w = image_bgr.shape[:2]
        size = INPAINT_SIZE
        small = cv2.resize(image_bgr, size, interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        blob = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = blob.transpose(2, 0, 1)[None] / 255.0
        mask_blob = (small_mask > 127).astype(np.float32)[None, None]
        out = self.session.run(None, {"image": blob, "mask": mask_blob})[0][0]
        filled = cv2.cvtColor(np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8),
                              cv2.COLOR_RGB2BGR)
        filled = cv2.resize(filled, (w, h), interpolation=cv2.INTER_LANCZOS4)
        feather = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 4)[:, :, None]
        return (filled.astype(np.float32) * feather
                + image_bgr.astype(np.float32) * (1.0 - feather)).astype(np.uint8)


# -- plate -----------------------------------------------------------------
def cover_resize(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fill the frame without letterboxing, cropping the longer side."""
    h, w = image_bgr.shape[:2]
    if (w, h) == (width, height):
        return image_bgr.copy()
    scale = max(width / w, height / h)
    resized = cv2.resize(image_bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                         interpolation=cv2.INTER_AREA)
    x = (resized.shape[1] - width) // 2
    y = (resized.shape[0] - height) // 2
    return resized[y:y + height, x:x + width].copy()


def person_mask(alpha: np.ndarray) -> np.ndarray:
    """The reference person, grown enough to take their hair and shadow edge."""
    mask = (alpha > 0.30).astype(np.uint8) * 255
    if not mask.any():
        return mask
    return cv2.dilate(mask, np.ones((17, 17), np.uint8), iterations=1)


def flatten_halo(image: np.ndarray, mask: np.ndarray, radius: int = 64,
                 strength: float = 0.92) -> np.ndarray:
    """Even out the pale seam that inpainting leaves around the old silhouette.

    LaMa reconstructs the wall well but its output is a touch brighter and
    softer than the photo, so the place where the reference person's hair was
    keeps a faint glow. A Gaussian average of the wall is indistinguishable from
    the wall itself, so only low-detail pixels near the mask are flattened; the
    artwork, shelf and chair keep their edges.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 9)
    detail = cv2.absdiff(image, blurred).max(axis=2).astype(np.float32)
    smooth = np.clip((16.0 - detail) / 10.0, 0.0, 1.0)
    distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ring = np.clip(1.0 - distance / float(radius), 0.0, 1.0)
    weight = (ring * smooth * strength)[:, :, None]
    return (image.astype(np.float32) * (1.0 - weight)
            + blurred.astype(np.float32) * weight).astype(np.uint8)


def _plate_cache_path(path: str | Path, width: int, height: int) -> Path:
    from .config import app_data_dir

    try:
        digest = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    folder = app_data_dir() / "studio"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}-{width}x{height}-{PLATE_VERSION}.png"


def is_plate_cached(reference_path: str | Path, width: int, height: int) -> bool:
    try:
        cache = _plate_cache_path(reference_path, width, height)
    except OSError:
        return False
    return cache.exists() and cache.stat().st_size > 1024


def load_plate(
    reference_path: str | Path,
    width: int,
    height: int,
    matte_model: Optional[matte.MatteModel] = None,
    inpaint_model: Optional[InpaintModel] = None,
    on_log: Optional[Callable[[str], None]] = None,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    """The reference room without its person, plus the seat's face box.

    The cleaned plate is cached next to the character images so the one-second
    inpaint happens once per photo, not once per launch.
    """
    log = on_log or (lambda _m: None)
    cache = _plate_cache_path(reference_path, width, height)
    source = face.load_image(reference_path)
    plate = cover_resize(source, width, height)

    detector = face.FaceDetector(_detector_path(), max_side=960)
    reference_face = face.largest(detector.detect(plate))
    face_box = reference_face.box if reference_face else None

    if cache.exists():
        cleaned = cv2.imdecode(np.fromfile(str(cache), np.uint8), cv2.IMREAD_COLOR)
        if cleaned is not None and cleaned.shape[:2] == (height, width):
            log(f"Studio plate loaded from cache ({cache.name})")
            return cleaned, face_box

    if matte_model is None or inpaint_model is None:
        raise StudioError("the background models are not loaded")

    alpha = matte_model.alpha(plate)
    mask = person_mask(alpha)
    coverage = float((mask > 0).mean())
    if coverage < 0.01:
        log("Reference image has no person in it - using the photo as the plate")
        cleaned = plate
    else:
        log(f"Removing the reference person ({coverage * 100:.0f}% of the frame)")
        cleaned = inpaint_model.fill(plate, mask)
        cleaned = flatten_halo(cleaned, mask)

    try:
        cv2.imencode(".png", cleaned)[1].tofile(str(cache))
    except (OSError, cv2.error):
        pass  # a cache miss next launch is a second, not a failure
    return cleaned, face_box


def _detector_path() -> str:
    from . import models

    return str(models.path("detector"))


# -- composite -------------------------------------------------------------
def ambient_lab(plate: np.ndarray) -> np.ndarray:
    """Average colour of the room, ignoring the very dark corners.

    The corners of a photo like this are usually the unlit edge of a shelf; the
    light that would fall on a person comes from the wall and desk area.
    """
    h, w = plate.shape[:2]
    patch = plate[int(h * 0.08):int(h * 0.62), int(w * 0.06):int(w * 0.94)]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3)
    weights = np.clip((lab[:, 0] - 12.0) / 60.0, 0.0, 1.0)
    total = float(weights.sum())
    if total < 1e-3:
        return lab.mean(axis=0)
    return (lab * weights[:, None]).sum(axis=0) / total


@dataclass
class Seat:
    scale: float
    tx: float
    ty: float

    def matrix(self) -> np.ndarray:
        return np.array([[self.scale, 0.0, self.tx],
                         [0.0, self.scale, self.ty]], dtype=np.float32)


class StudioScene:
    """Holds the cleaned room and composites each frame into it."""

    def __init__(self, plate: np.ndarray, reference_face: Optional[tuple[int, int, int, int]] = None,
                 on_log: Optional[Callable[[str], None]] = None) -> None:
        self.plate = plate
        self.height, self.width = plate.shape[:2]
        self.ambient = ambient_lab(plate)
        self.anchor_x = float(self.width / 2)
        self.reference_face_width = 0.0
        if reference_face is not None:
            x, _y, w, _h = reference_face
            self.anchor_x = x + w / 2.0
            self.reference_face_width = float(w)
        self.seat: Optional[Seat] = None
        self._log = on_log or (lambda _m: None)
        # The matte arrives on its own thread and the composite reuses it for
        # several frames, so the shadow only needs rebuilding when the alpha it
        # was derived from actually changes. Keyed on the array itself: the
        # reference keeps it alive, so the identity cannot be recycled. The
        # cached value is the multiplier itself (255 = untouched plate), which
        # saves a stack of per-frame float maths over the whole paste region.
        self._shadow_cache: Optional[tuple[np.ndarray, tuple, tuple, np.ndarray]] = None

    # -- seating -----------------------------------------------------------
    def _measure(self, face_box: tuple[int, int, int, int]) -> Seat:
        x, _y, w, _h = face_box
        if self.reference_face_width > 0 and w > 0:
            scale = self.reference_face_width / float(w) * SEAT_GAIN
        else:
            scale = 1.18
        scale = float(np.clip(scale, MIN_SCALE, MAX_SCALE))
        tx = self.anchor_x - scale * (x + w / 2.0)
        if scale >= 1.0:
            # Magnified: the source must cover the output, or an arm that reaches
            # the camera edge would show the end of the frame.
            tx = float(np.clip(tx, self.width * (1.0 - scale), 0.0))
        else:
            # A close-up camera needs scale < 1 to fit the reference seat. That
            # leaves room at the sides of the source frame, which is fine: alpha
            # is zero there and the plate shows through. Only keep the face near
            # the middle so an off-centre camera cannot push the subject out.
            face_cx = scale * (x + w / 2.0)
            tx = float(np.clip(tx, self.width * 0.24 - face_cx,
                               self.width * 0.76 - face_cx))
        # Anchored at the bottom: the camera cut the torso off there, so the cut
        # is hidden by the frame edge. Lifting the person to the reference photo's
        # eye height would need invented torso, and replicated shirt rows read as
        # a flat slab.
        ty = float(self.height * (1.0 - scale))
        return Seat(scale=scale, tx=tx, ty=ty)

    def seat_person(self, face_box: tuple[int, int, int, int]) -> Seat:
        if self.seat is None:
            self.seat = self._measure(face_box)
            self._log(
                f"Seated in the reference room: scale {self.seat.scale:.2f}, "
                f"offset ({self.seat.tx:.0f}, {self.seat.ty:.0f})"
            )
        return self.seat

    def reset(self) -> None:
        self.seat = None

    # -- per-frame ---------------------------------------------------------
    def composite(self, frame_bgr: np.ndarray, alpha: np.ndarray,
                  face_box: Optional[tuple[int, int, int, int]] = None,
                  edge_color: Optional[np.ndarray] = None) -> np.ndarray:
        """Put the person into the room. ``alpha`` is a refined soft matte.

        ``edge_color`` is the same frame with its soft rim already replaced by
        the person's own nearby colour (see :func:`mimiccam.matte.decontaminate`).
        Swapping it in before the warp removes the pale outline that camera
        pixels behind a semi-transparent edge otherwise leave on hair.
        """
        if face_box is not None and self.seat is None:
            self.seat_person(face_box)
        if self.seat is None:
            return self.plate.copy()

        height, width = frame_bgr.shape[:2]
        if (width, height) != (self.width, self.height):
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height),
                                   interpolation=cv2.INTER_AREA)
            alpha = cv2.resize(alpha, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        # One scan finds both the person's bounds (for the warp) and whether
        # there is anybody in frame at all. Scanning every other pixel and
        # doubling the result costs a quarter of the full pass (0.8 ms against
        # 2.8 ms at 720p) and lands within the 8 px pad anyway.
        rows, cols = np.nonzero(alpha[::2, ::2] > 0.02)
        if rows.size == 0:
            return self.plate.copy()
        box = (max(0, int(cols.min()) * 2 - 8), max(0, int(rows.min()) * 2 - 8),
               min(width, int(cols.max()) * 2 + 10), min(height, int(rows.max()) * 2 + 10))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return self.plate.copy()

        if edge_color is not None and edge_color.shape[:2] == frame_bgr.shape[:2]:
            x0, y0, x1, y1 = box
            rim = alpha[y0:y1, x0:x1]
            # cv2.compare writes the 0/255 mask directly; the boolean version
            # built three whole-plane temporaries and cost 2.5 ms.
            band = cv2.bitwise_and(cv2.compare(rim, 0.02, cv2.CMP_GT),
                                   cv2.compare(rim, 0.985, cv2.CMP_LT))
            # The rim colour is already the person's, not the room's, so it goes
            # in at full strength. Only the rim is touched, with a SIMD blit
            # rather than boolean-index assignment.
            cv2.copyTo(edge_color[y0:y1, x0:x1], band, frame_bgr[y0:y1, x0:x1])

        person, person_alpha, origin = self._warp(frame_bgr, alpha, self.seat.matrix(), box)
        if person is None:
            return self.plate.copy()

        person = self._relight(person, person_alpha)
        return self._blend(person, person_alpha, origin)

    def _blend(self, person: np.ndarray, alpha: np.ndarray,
               origin: tuple[int, int]) -> np.ndarray:
        """Shadow and paste inside one region, never two full-frame float planes.

        A full 720p float composite costs ~40 ms on this laptop; restricting the
        work to the person plus a shadow margin costs about a third of that and
        is pixel-identical everywhere it matters.
        """
        x, y = origin
        ph, pw = alpha.shape[:2]
        pad = 56
        rx0, ry0 = max(0, x - pad), max(0, y - pad)
        rx1, ry1 = min(self.width, x + pw + pad), min(self.height, y + ph + pad)
        if rx1 - rx0 < 8 or ry1 - ry0 < 8:
            return self.plate.copy()

        # One plate copy, then everything happens in views of it: copying the
        # region out and back cost ~6 ms at this size.
        out = self.plate.copy()
        region = out[ry0:ry1, rx0:rx1]
        px, py = x - rx0, y - ry0
        factor = self._cached_shadow(alpha, (px, py), region.shape[:2])
        if factor is not None:
            cv2.multiply(region, factor, dst=region, scale=1.0 / 255.0)

        target = region[py:py + ph, px:px + pw]
        weight = cv2.convertScaleAbs(alpha, alpha=255.0)
        weight = cv2.cvtColor(weight, cv2.COLOR_GRAY2BGR)
        inverse = cv2.bitwise_not(weight)
        # uint8 SIMD blending: a quarter of the float-plane cost on this laptop,
        # and 1/255 quantisation is invisible in a video frame.
        front = cv2.multiply(person, weight, scale=1.0 / 255.0)
        back = cv2.multiply(target, inverse, scale=1.0 / 255.0)
        cv2.add(front, back, dst=target)
        return out

    def _warp(self, frame: np.ndarray, alpha: np.ndarray, matrix: np.ndarray,
              box: Optional[tuple[int, int, int, int]] = None):
        """Warp only the person's bounding box, not two full 720p planes."""
        h, w = frame.shape[:2]
        if box is None:
            ys, xs = np.where(alpha > 0.02)
            if ys.size == 0:
                return None, alpha, (0, 0)
            pad = 8
            x0 = max(0, int(xs.min()) - pad)
            y0 = max(0, int(ys.min()) - pad)
            x1 = min(w, int(xs.max()) + pad)
            y1 = min(h, int(ys.max()) + pad)
        else:
            x0, y0, x1, y1 = box
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, alpha, (0, 0)

        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        projected = cv2.transform(corners.reshape(1, -1, 2), matrix).reshape(-1, 2)
        dx0 = max(0, int(np.floor(projected[:, 0].min())))
        dy0 = max(0, int(np.floor(projected[:, 1].min())))
        dx1 = min(self.width, int(np.ceil(projected[:, 0].max())))
        dy1 = min(self.height, int(np.ceil(projected[:, 1].max())))
        if dx1 - dx0 < 8 or dy1 - dy0 < 8:
            return None, alpha, (0, 0)

        # The warp input is the source *crop*, so its coordinates start at 0,0:
        # translate to full-frame coordinates before the seat transform, then
        # translate the result to the destination ROI.
        pre = np.eye(3, dtype=np.float32)
        pre[0, 2] = x0
        pre[1, 2] = y0
        shift = np.eye(3, dtype=np.float32)
        shift[0, 2] = -dx0
        shift[1, 2] = -dy0
        roi_matrix = (shift @ np.vstack([matrix, [0.0, 0.0, 1.0]]) @ pre)[:2]
        size = (dx1 - dx0, dy1 - dy0)
        warped = cv2.warpAffine(frame[y0:y1, x0:x1], roi_matrix, size,
                                flags=cv2.INTER_LINEAR)
        warped_alpha = cv2.warpAffine(alpha[y0:y1, x0:x1], roi_matrix, size,
                                      flags=cv2.INTER_LINEAR)
        return warped, np.clip(warped_alpha, 0.0, 1.0), (dx0, dy0)

    def _relight(self, person: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """Pull the person's colour towards the room's light, at full detail.

        The room is warmer and dimmer than a webcam lit by a monitor, and a face
        that keeps its own white balance reads as pasted. Chroma moves further
        than luminance on purpose: losing exposure detail is worse than a small
        colour mismatch.

        Applied through one 3-channel lookup table in CIELAB, which is exactly
        what the correction is - a constant chroma shift plus a luminance
        curve - so it costs two colour conversions and no float planes. The
        correction used to be built on a quarter-size copy and interpolated
        back, which halved the sharpness of everything pasted into the room
        (Laplacian variance 589 -> 13 on a test crop).
        """
        if float(alpha.sum()) < 256.0:
            return person
        # Statistics from a strided sample: ~20k pixels, no resize of the image.
        small = np.ascontiguousarray(person[::4, ::4])
        small_alpha = alpha[::4, ::4]
        sample = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        weights = np.clip(small_alpha, 0.0, 1.0).reshape(-1)
        if float(weights.sum()) < 64.0:
            return person
        current = (sample.reshape(-1, 3) * weights[:, None]).sum(axis=0) / float(weights.sum())

        da = float((self.ambient[1] - current[1]) * 0.55)
        db = float((self.ambient[2] - current[2]) * 0.55)
        # Exposure: a backlit webcam leaves the person far darker than the room
        # they are being placed into. Match most of the way, lifting shadows more
        # than highlights so a dark shirt gains detail instead of turning grey
        # and the face does not clip. Never darken by more than a hair: the
        # source is the real exposure, the room is the target.
        target = float(np.clip(self.ambient[0] * 0.85, 20.0, 200.0))
        ratio = float(np.clip(target / max(1.0, current[0]), 0.92, 1.7))

        levels = np.arange(256, dtype=np.float32)
        # Hair, eyebrows and a black shirt must stay dark: lifting a near-black
        # pixel by the room's exposure turns the hairline grey. The room's lamps
        # light surfaces, not silhouettes, so the correction fades out as the
        # source approaches black and keeps a small share everywhere else.
        share = 0.25 + 0.75 * np.clip((levels - 20.0) / 50.0, 0.0, 1.0)
        lut = np.empty((1, 256, 3), dtype=np.uint8)
        lut[0, :, 0] = np.clip(levels * (1.0 + (ratio - 1.0) * share), 0, 255)
        lut[0, :, 1] = np.clip(levels + da, 0, 255)
        lut[0, :, 2] = np.clip(levels + db, 0, 255)
        return cv2.cvtColor(cv2.LUT(cv2.cvtColor(person, cv2.COLOR_BGR2LAB), lut),
                            cv2.COLOR_LAB2BGR)

    def _cached_shadow(self, alpha: np.ndarray, offset: tuple[int, int],
                       shape: tuple[int, int]) -> Optional[np.ndarray]:
        """Plate multiplier for this alpha, rebuilt only when the alpha is new.

        Returns a 3-channel uint8 map where 255 leaves the plate alone, ready
        for ``cv2.multiply(region, factor, scale=1/255)``.
        """
        cache = self._shadow_cache
        if cache is not None and cache[0] is alpha and cache[1] == offset and cache[2] == shape:
            return cache[3]
        field = self._shadow_roi(alpha, offset, shape)
        if field is None:
            return None
        # Darken by at most SHADOW_STRENGTH, and only where the shadow is.
        factor = cv2.convertScaleAbs(field, alpha=-255.0 * SHADOW_STRENGTH, beta=255.0)
        factor = cv2.cvtColor(factor, cv2.COLOR_GRAY2BGR)
        self._shadow_cache = (alpha, offset, shape, factor)
        return factor

    def _shadow_roi(self, alpha: np.ndarray, offset: tuple[int, int],
                    shape: tuple[int, int]) -> Optional[np.ndarray]:
        """Soft cast shadow for one region, at half resolution.

        A shadow is a smooth field, so the extra precision of a 720p Gaussian
        buys nothing; blurring the region at half size costs ~1 ms.
        """
        height, width = shape[:2]
        x, y = offset
        if height < 16 or width < 16:
            return None
        hw, hh = max(4, width // 2), max(4, height // 2)
        small = cv2.resize(alpha, (max(1, alpha.shape[1] // 2), max(1, alpha.shape[0] // 2)),
                           interpolation=cv2.INTER_AREA)
        x0, y0 = max(0, x // 2), max(0, y // 2)
        field = np.zeros((hh, hw), dtype=np.float32)
        ch = min(small.shape[0], hh - y0)
        cw = min(small.shape[1], hw - x0)
        if ch <= 0 or cw <= 0:
            return None
        field[y0:y0 + ch, x0:x0 + cw] = small[:ch, :cw]
        # Two box passes approximate the sigma-9 Gaussian at a third of the
        # cost: a shadow is a gradient, not a shape, so the slight change in
        # profile is invisible.
        blurred = cv2.blur(cv2.blur(field, (13, 13)), (13, 13))
        # A tight ambient-occlusion ring just outside the silhouette. It grounds
        # the person, and it also hides the last 1-2 px of camera background
        # that a soft matte always leaves at the edge. Kept weak on purpose: at
        # 2.4x it painted a dark outline around the whole head, which reads as a
        # cutout rather than as a lit surface.
        ring = np.clip(cv2.GaussianBlur(field, (0, 0), 2) - field, 0.0, 1.0) * 1.1
        # Slide the shadow down: light comes from above, so the contact shadow
        # sits under the shoulders rather than behind the head. Slice, don't
        # roll: a roll wraps the bottom edge onto the ceiling.
        shifted = np.zeros_like(blurred)
        shifted[8:] = blurred[:-8]
        shifted = np.maximum(shifted, np.clip(ring, 0.0, 1.0))
        return cv2.resize(shifted, (width, height), interpolation=cv2.INTER_LINEAR)
