# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Wear the character photo's clothes, not just their face.

The body pass moves skin tone; it deliberately leaves clothing alone, because
recolouring a shirt without knowing where the shirt ends produces blotches. This
is the other half: the character photo already contains a dressed torso, so the
cheapest honest "body swap" is to *transplant that garment* rather than invent
one. No generative model, no extra download, and it runs on the same matte the
room composite already uses.

How it works:

1. At setup, the reference photo is matted (RVM) and the torso is cut out of it:
   everything below the chin, minus the face and hair, minus skin. What is left
   is cloth - jacket, shirt, tie, whatever the character is wearing.
2. Per frame, the reference torso quad (shoulders and waist, read off its own
   matte) is mapped onto the live torso quad by an affine fit, so leaning or
   shifting moves the clothes with the body. When the live silhouette below the
   chin is unusable - a dark room, an arm across the chest, nobody there - it
   falls back to scaling by the ratio of the two face widths and anchoring the
   chins together, which is the ruler the rest of the engine already uses (see
   ``studio.Seat``).
3. The warp is composited inside the live person's matte, with the live skin
   detector punched back through, so hands, forearms and neck stay yours. The
   body pass then tones that surviving skin to the swapped face.
4. Exposure is left as the reference shot it. Two automatic relights were tried
   and both were worse than doing nothing: matching the garment to the room's
   ambient crushed a navy suit to black in a backlit room (the same room that
   hides a face, see ``exposure.py``), and matching it to the shirt underneath
   turned the character's dark suit white because the shirt was white. Saying so
   here is cheaper than rediscovering it.

What this cannot do, and does not pretend to: it does not change your body.
The quad tracks the silhouette, not the skeleton, so a raised arm carries the
sleeve with the silhouette's outline rather than bending the elbow; a turn
beyond about 30 degrees squashes a flat patch in a way a photo cannot hide.
Real garment synthesis with sleeves that follow a raised arm is a generative
job; this is the part that can be done honestly at 30 fps on this machine.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

from . import body

# Where the cloth starts, in face heights above the chin. Anything higher eats
# the jaw, and the face paste lands on top of it anyway.
TOP_ABOVE_CHIN = 0.10
# Fraction of the reference face width the torso must occupy before this is
# worth doing at all. A portrait cropped at the collarbone has nothing to wear.
MIN_COVERAGE = 0.02
# How much of the live person's own skin stays in front of the garment.
SKIN_KEEP = 0.90
# Edge feather, as a fraction of the patch width.
SEAM = 0.02
# The patch may be needed at a very different scale from the reference framing.
MIN_SCALE, MAX_SCALE = 0.25, 3.0
# How far the dye reaches around the live face, in face widths / heights. The
# live shoulders are usually wider than the reference's, and a white shirt left
# showing beside a pasted suit is what makes the transplant look glued on.
TORSO_HALF_WIDTHS = 2.6
TORSO_DEPTH = 2.8
# Fraction of the patch height over which the garment fades in at the top.
TOP_FEATHER = 0.10
# Torso tracking. The garment is warped from the character's torso quad onto the
# live torso quad, so leaning, turning or shifting moves the clothes with the
# body instead of pinning them to the face. The quad's two rows are measured
# below the chin in *face heights*, the same ruler for both photos: the bottom of
# a matte is an arbitrary crop (the character photo goes to the knees, a webcam
# frame stops at the waist), so using it would squash the garment by a factor
# that says nothing about either body.
#
# A silhouette narrower than this fraction of the frame is not a torso (an arm, a
# chair, a shadow) and the face-width transform takes over.
QUAD_MIN_WIDTH = 0.12
# Weight of the newest quad when smoothing, so one bad silhouette does not throw
# the clothes sideways for a frame.
QUAD_SMOOTH = 0.55
# The two rows, in face heights below the chin: shoulder line and mid-torso.
QUAD_TOP_DEPTH = 0.20
QUAD_DEPTH = 1.60
# How far to search outwards when the matte has a hole at a target row (a gap
# under the jaw, an arm across the waist), in face heights.
QUAD_SEARCH = 0.40


def torso_quad(alpha: np.ndarray, face_box: tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Four corners of the torso from the person matte: shoulders and waist.

    Returns ``[top-left, top-right, bottom-right, bottom-left]`` in image
    coordinates, or None when the silhouette below the chin is too thin to be a
    torso. This is the anchor that makes the garment follow the body rather than
    the head: the reference's quad is mapped onto the live quad by an affine fit,
    so a lean or a shift drags the clothes with it.

    Both rows are placed in face heights below the chin (see ``QUAD_DEPTH``), so
    the same rule picks comparable rows in a full-length portrait and a webcam
    frame.
    """
    height, width = np.shape(alpha)[:2]
    x, y, w, h = (int(v) for v in face_box)
    if w <= 2 or h <= 2:
        return None
    chin = min(height - 1, max(0, y + h))
    solid = np.asarray(alpha) > 0.5
    if int(solid[chin:].sum()) < 500:
        return None
    band = max(2, int(h * 0.12))
    reach = max(band, int(h * QUAD_SEARCH))

    def extent(row: int) -> Optional[tuple[float, float]]:
        """Left and right edge of the silhouette near ``row``.

        Searches outwards when the matte is missing there: the hole can be above
        (a gap under the jaw) or below (an arm across the waist).
        """
        step = max(1, band // 2)
        offsets = [0]
        for shift in range(step, reach + 1, step):
            offsets.extend((shift, -shift))
        for shift in offsets:
            centre = row - shift
            lo = max(0, centre - band // 2)
            hi = min(height, centre + band // 2 + 1)
            block = solid[lo:hi]
            if block.size == 0 or not block.any():
                continue
            columns = np.flatnonzero(block.any(axis=0))
            return float(columns[0]), float(columns[-1])
        return None

    top = extent(min(height - 1, chin + max(2, int(h * QUAD_TOP_DEPTH))))
    bottom_row = min(height - 1, chin + max(4, int(h * QUAD_DEPTH)))
    bottom = extent(bottom_row)
    if top is None or bottom is None:
        return None
    widest = max(top[1] - top[0], bottom[1] - bottom[0])
    if widest < QUAD_MIN_WIDTH * width:
        return None
    # The top edge is the chin itself, not the first solid row of the matte: a
    # gap under the jaw (a hand, a shadow) would otherwise push the collar down
    # the chest. Widths still come from the matte below it.
    return np.array(
        [[top[0], float(chin)], [top[1], float(chin)],
         [bottom[1], float(bottom_row)], [bottom[0], float(bottom_row)]],
        np.float32,
    )


def _affine_fit(source: np.ndarray, target: np.ndarray) -> Optional[np.ndarray]:
    """Least-squares affine (3x3) mapping the source quad corners onto the target.

    A homography is the exact answer for four points, and the wrong tool here: the
    garment patch runs well past its anchor quad, and a projective map
    extrapolated beyond its own corners explodes. On the real photo-and-webcam
    pair the homography put the patch's bottom row 400 px outside the frame and
    the denominator fell to 0.2 of unity, which is a warp of the clothes rather
    than a change of them. An affine fit keeps the parts that carry a lean -
    scale, shift, shear - and stays bounded wherever the patch reaches.
    """
    source = np.asarray(source, np.float64)
    target = np.asarray(target, np.float64)
    if source.shape != target.shape or source.size != 8:
        return None
    design, rhs = [], []
    for (sx, sy), (tx, ty) in zip(source, target):
        design.append([sx, sy, 1.0, 0.0, 0.0, 0.0])
        rhs.append(tx)
        design.append([0.0, 0.0, 0.0, sx, sy, 1.0])
        rhs.append(ty)
    try:
        solution, *_ = np.linalg.lstsq(np.asarray(design), np.asarray(rhs), rcond=None)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate quad
        return None
    a, b, tx, c, d, ty = (float(v) for v in solution)
    determinant = a * d - b * c
    if not np.isfinite(determinant) or determinant <= 0.0:
        return None
    return np.array([[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]], np.float32)


def torso_mask(image_bgr: np.ndarray, face_box: tuple[int, int, int, int],
               alpha: np.ndarray) -> np.ndarray:
    """Cloth on the reference person: person alpha, below the chin, no skin."""
    x, y, w, h = (int(v) for v in face_box)
    mask = np.asarray(alpha, dtype=np.float32).copy()
    chin = y + h
    mask[:max(0, chin - int(h * TOP_ABOVE_CHIN))] = 0.0
    mask *= 1.0 - np.clip(body.skin_mask(image_bgr) * 1.2, 0.0, 1.0)
    mask[mask < 0.35] = 0.0
    return mask



class Wardrobe:
    """Holds the character's garment and pastes it on each frame."""

    def __init__(self, reference_bgr: np.ndarray, face_box: tuple[int, int, int, int],
                 matte_model, on_log: Optional[Callable[[str], None]] = None) -> None:
        self._log = on_log or (lambda _m: None)
        self.face_box = tuple(int(v) for v in face_box)
        self.patch: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.origin = (0, 0)
        self.quad: Optional[np.ndarray] = None

        alpha = matte_model.alpha(reference_bgr)
        self.quad = torso_quad(alpha, self.face_box)
        self._live_quad: Optional[np.ndarray] = None
        mask = torso_mask(reference_bgr, self.face_box, alpha)
        coverage = float(mask.mean())
        if coverage < MIN_COVERAGE:
            self._log("Wardrobe skipped: the character photo shows no torso below the chin")
            return

        rows, cols = np.nonzero(mask > 0)
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        self.patch = reference_bgr[y0:y1, x0:x1].copy()
        self.mask = mask[y0:y1, x0:x1].copy()
        self.origin = (x0, y0)
        # Feather the outline so the garment does not end in a cut line, and ramp
        # the top: the crop's first rows are a straight cut across the chest,
        # which showed as a hard horizontal seam where the reference is wider
        # than the live person.
        k = max(3, int(min(self.mask.shape[:2]) * SEAM * 2) | 1)
        self.mask = cv2.GaussianBlur(self.mask, (k, k), 0)
        ramp = max(2, int(self.mask.shape[0] * TOP_FEATHER))
        fade = np.linspace(0.0, 1.0, ramp, dtype=np.float32)[:, None]
        self.mask[:ramp] *= fade
        follow = " - fitted to your shoulders and waist" if self.quad is not None else ""
        self._log(
            f"Wardrobe ready: {x1 - x0}x{y1 - y0} px of the character's clothes "
            f"({coverage * 100:.0f}% of the photo){follow}"
        )

    @property
    def ready(self) -> bool:
        return self.patch is not None and self.mask is not None

    def matrix_for(self, live_face_box: tuple[int, int, int, int]) -> tuple[np.ndarray, float]:
        """Scale by face width, anchor the chins together."""
        x, y, w, h = (int(v) for v in live_face_box)
        rx, ry, rw, rh = self.face_box
        scale = float(np.clip(w / max(1.0, float(rw)), MIN_SCALE, MAX_SCALE))
        ref_cx, ref_chin = rx + rw / 2.0, ry + rh
        live_cx, live_chin = x + w / 2.0, y + h
        px, py = self.origin
        matrix = np.array(
            [[scale, 0.0, live_cx + (px - ref_cx) * scale],
             [0.0, scale, live_chin + (py - ref_chin) * scale]],
            np.float32,
        )
        return matrix, scale

    def transform_for(self, live_face_box: tuple[int, int, int, int],
                      person_alpha: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """3x3 map from the patch to this frame.

        Preference order: the reference torso quad onto the live torso quad, so
        the garment follows a lean or a shift instead of hanging in space, and
        the face-width/chin transform when the silhouette is unusable (dark
        room, out of frame, an arm across the chest).
        """
        fallback = np.eye(3, dtype=np.float32)
        fallback[:2] = matrix
        if self.quad is None:
            return fallback
        live = torso_quad(person_alpha, live_face_box)
        if live is None:
            self._live_quad = None
            return fallback
        previous = self._live_quad
        # Smooth: a single bad silhouette should not throw the clothes sideways
        # before the next frame corrects it.
        self._live_quad = (
            live if previous is None else QUAD_SMOOTH * live + (1.0 - QUAD_SMOOTH) * previous
        )
        px, py = self.origin
        source = (self.quad - np.array([px, py], np.float32)).astype(np.float32)
        transform = _affine_fit(source, self._live_quad)
        if transform is None:
            return fallback
        # Sanity-bound the fit against the face-width scale rather than against an
        # absolute one: the reference photo is 3356 px tall and the frame is 720,
        # so a healthy torso mapping runs at a small multiple of the face ratio.
        # A silhouette that is one arm or a chair is not within it.
        linear = transform[:2, :2]
        scale = float(np.sqrt(abs(np.linalg.det(linear))))
        expected = float(matrix[0, 0])
        if not (0.3 * expected <= scale <= 3.0 * expected):
            return fallback
        return transform

    def dress(self, frame_bgr: np.ndarray, live_face_box: tuple[int, int, int, int],
              person_alpha: np.ndarray) -> np.ndarray:
        """Composite the garment into this frame, or return it untouched."""
        if not self.ready:
            return frame_bgr
        height, width = frame_bgr.shape[:2]
        matrix, _scale = self.matrix_for(live_face_box)
        transform = self.transform_for(live_face_box, person_alpha, matrix)

        # Warp only the patch's destination box: a full 720p warp of the garment
        # and its alpha costs three times what the region does.
        ph, pw = self.mask.shape[:2]
        corners = cv2.perspectiveTransform(
            np.array([[[0.0, 0.0], [pw, 0.0], [0.0, ph], [pw, ph]]], np.float32), transform
        )[0]
        x0 = max(0, int(np.floor(corners[:, 0].min())) - 2)
        y0 = max(0, int(np.floor(corners[:, 1].min())) - 2)
        x1 = min(width, int(np.ceil(corners[:, 0].max())) + 3)
        y1 = min(height, int(np.ceil(corners[:, 1].max())) + 3)
        # Widen the work region to the live torso, not just the garment: the dye
        # below has to reach shoulders the reference's own outline does not.
        lx, ly, lw, lh = (int(v) for v in live_face_box)
        half = int(lw * TORSO_HALF_WIDTHS)
        x0 = min(x0, max(0, lx + lw // 2 - half))
        x1 = max(x1, min(width, lx + lw // 2 + half))
        y1 = max(y1, min(height, int(ly + lh * TORSO_DEPTH)))
        # Reach above the garment so the dye can fade in instead of starting on a
        # straight line across the chest.
        y0 = max(0, y0 - int(0.18 * (y1 - y0)))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return frame_bgr
        roi = (x0, y0, x1, y1)
        local = transform.copy()
        local[0, 0] -= local[2, 0] * x0
        local[0, 1] -= local[2, 1] * x0
        local[0, 2] -= local[2, 2] * x0
        local[1, 0] -= local[2, 0] * y0
        local[1, 1] -= local[2, 1] * y0
        local[1, 2] -= local[2, 2] * y0
        size = (x1 - x0, y1 - y0)

        garment = cv2.warpPerspective(self.patch, local, size, flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        weight = cv2.warpPerspective(self.mask, local, size, flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        weight = np.clip(weight, 0.0, 1.0)

        # Crop the matte to the same region: copying the *whole* alpha in here
        # squashes the frame into the ROI and the mask lines up with nothing.
        person = np.clip(person_alpha[y0:y1, x0:x1], 0.0, 1.0)
        skin = cv2.resize(body.skin_mask(frame_bgr[y0:y1, x0:x1]),
                          (size[0], size[1]), interpolation=cv2.INTER_AREA)
        clothing = person * np.clip(1.0 - skin * SKIN_KEEP, 0.0, 1.0)
        blend = weight * clothing
        out = self._composite(frame_bgr, garment, blend, roi)
        # The transplant only reaches as far as the reference's own outline: a
        # live person wider than the character kept their own shirt at the sides,
        # which read as two white bands around a floating suit. Dye the rest of
        # the clothing toward the garment's colour so the torso reads as one
        # outfit, and let the transplanted cloth keep its lapels and tie on top.
        target = body.mean_lab(garment, weight)
        if target is not None:
            out = self._dye(out, roi, clothing * (1.0 - weight), target)
        return out

    @staticmethod
    def _dye(frame: np.ndarray, roi: tuple[int, int, int, int], mask: np.ndarray,
             target_lab: np.ndarray, strength: float = 0.85) -> np.ndarray:
        """Move the remaining clothing's colour toward the garment's."""
        x0, y0, x1, y1 = roi
        if float(mask.max()) < 0.15:
            return frame
        height, width = y1 - y0, x1 - x0
        # Fade the top edge: the dye has to arrive, not begin on a straight line.
        ramp = max(2, int(height * TOP_FEATHER * 2))
        mask = np.array(mask, dtype=np.float32, copy=True)
        mask[:ramp] *= np.linspace(0.0, 1.0, ramp, dtype=np.float32)[:, None]
        small_w, small_h = max(2, width // 2), max(2, height // 2)
        region = frame[y0:y1, x0:x1]
        small = cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
        weight = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
        current = body.mean_lab(small, weight)
        if current is None:
            return frame
        # Apply the full correction on the small grid, then lerp it in at full
        # resolution with the mask. Writing the small result straight back would
        # resample (and so blur) the pixels the mask excludes.
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1:] += (np.asarray(target_lab, np.float32)[1:] - current[1:])[None, None, :]
        ratio = float(np.clip(float(target_lab[0]) / max(1.0, float(current[0])), 0.6, 1.5))
        lab[:, :, 0] *= ratio
        shifted = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        shifted_full = cv2.resize(shifted, (width, height), interpolation=cv2.INTER_LINEAR)

        weight_full = cv2.convertScaleAbs(np.clip(mask, 0.0, 1.0) * strength, alpha=255.0)
        weight_full = cv2.cvtColor(weight_full, cv2.COLOR_GRAY2BGR)
        out = frame.copy()
        target = out[y0:y1, x0:x1]
        front = cv2.multiply(shifted_full, weight_full, scale=1.0 / 255.0)
        back = cv2.multiply(target, cv2.bitwise_not(weight_full), scale=1.0 / 255.0)
        cv2.add(front, back, dst=target)
        return out

    @staticmethod
    def _composite(frame: np.ndarray, garment: np.ndarray, blend: np.ndarray,
                   roi: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = roi
        out = frame.copy()
        target = out[y0:y1, x0:x1]
        weight = cv2.convertScaleAbs(blend, alpha=255.0)
        weight = cv2.cvtColor(weight, cv2.COLOR_GRAY2BGR)
        inverse = cv2.bitwise_not(weight)
        front = cv2.multiply(garment, weight, scale=1.0 / 255.0)
        back = cv2.multiply(target, inverse, scale=1.0 / 255.0)
        cv2.add(front, back, dst=target)
        return out
