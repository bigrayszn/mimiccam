"""Deterministic colour looks for the offline engine.

A swap network changes *who* is on camera. These looks change *how* the frame
is recorded, using ordinary image maths instead of a generative model, so they
cost a couple of milliseconds and never hallucinate. They are honest about what
they are: grading, not restyling. The generative presets (anime, cinematic) keep
their names in the UI but map to the closest deterministic treatment here.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

_vignettes: dict[tuple[int, int], np.ndarray] = {}


def _lut(points: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return np.clip(points(x), 0, 255).astype(np.uint8)


def neutral(frame_bgr: np.ndarray) -> np.ndarray:
    return frame_bgr


def _vignette(shape: tuple[int, int], strength: float = 0.18) -> np.ndarray:
    key = (shape[0], shape[1])
    cached = _vignettes.get(key)
    if cached is not None:
        return cached
    height, width = shape
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(xs * xs + ys * ys) / np.sqrt(2.0)
    falloff = (1.0 - strength * radius ** 2)[:, :, None]
    _vignettes[key] = falloff
    return falloff


def cinematic(frame_bgr: np.ndarray) -> np.ndarray:
    """Teal shadows, warm highlights, gentle S-curve, subtle vignette."""
    b, g, r = cv2.split(frame_bgr)
    r = cv2.LUT(r, _lut(lambda x: 1.10 * x + 6 + 18 * np.sin(np.pi * x / 255.0) ** 2))
    g = cv2.LUT(g, _lut(lambda x: 1.02 * x + 1))
    b = cv2.LUT(b, _lut(lambda x: 0.90 * x + 14 - 10 * (x / 255.0) ** 2))
    graded = cv2.merge((b, g, r))
    graded = cv2.convertScaleAbs(graded, alpha=1.06, beta=-6)
    graded = (graded.astype(np.float32) * _vignette(graded.shape[:2])).astype(np.uint8)
    return graded


def cel(frame_bgr: np.ndarray) -> np.ndarray:
    """Posterised, saturated, ink-outlined - a 2D animation feel."""
    height, width = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (max(1, width // 2), max(1, height // 2)),
                       interpolation=cv2.INTER_AREA)
    smooth = cv2.bilateralFilter(small, 7, 60, 60)
    smooth = cv2.resize(smooth, (width, height), interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1].astype(np.int16) * 1.35, 0, 255).astype(np.uint8)
    smooth = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Posterise to a handful of levels per channel: flat cel colour.
    levels = 5
    poster = (smooth.astype(np.uint8) // (256 // levels)) * (256 // levels)
    poster = poster.astype(np.uint8)

    edges = cv2.Canny(cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY), 55, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    ink = edges.astype(np.float32)[:, :, None] / 255.0
    return (poster.astype(np.float32) * (1.0 - 0.85 * ink)).astype(np.uint8)


def studio(frame_bgr: np.ndarray) -> np.ndarray:
    """Balanced studio-ish light: local contrast plus a soft sharpen."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    l = clahe.apply(l)
    balanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(balanced, (0, 0), 1.6)
    return cv2.addWeighted(balanced, 1.18, blurred, -0.18, 0)


LOOKS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "neutral": neutral,
    "cinematic": cinematic,
    "cel": cel,
    "studio": studio,
}

PRESET_LOOKS: dict[str, str] = {
    "character": "neutral",
    "identity": "studio",
    "cinematic": "cinematic",
    "anime": "cel",
    "suit": "neutral",
    "custom": "neutral",
}


def for_preset(preset_key: str) -> str:
    return PRESET_LOOKS.get(preset_key, "neutral")


def apply(frame_bgr: np.ndarray, look: str) -> np.ndarray:
    fn = LOOKS.get(look, neutral)
    return fn(frame_bgr)
