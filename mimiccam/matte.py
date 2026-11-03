"""Person matte for the studio background.

Robust Video Matting (MobileNetV3) is the only model here that is *built* for
video: it carries four recurrent state tensors between frames, so the alpha it
returns does not flicker on hair or on a still shoulder the way a frame-by-frame
segmenter does. The network runs at 448x252 on the GPU (about 5 ms through
DirectML) and the soft alpha is upscaled to the frame.

Two post-steps matter for compositing:

``refine``   removes the last bright rim of the camera background from the edge
             of the matte. A soft alpha still contains the old background in
             partially covered pixels; pasting that onto a dim room puts a
             glowing outline around the person.
``decontaminate``
             replaces the colour under the soft rim with the person's own
             nearby colour, so a black hairline over a bright wall does not
             keep a pale fringe when it is pasted onto a dim room.

No green screen, no depth sensor, no manual masking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# 512x288 measured 7 ms on the laptop GPU; 384 is faster but the hairline
# blocks when upscaled. The alpha is refined at this size and then snapped to
# the full-resolution colour guide, which recovers the real edge.
MATTE_SIZE = (448, 252)
# How much of the frame the encoder actually reads. This, not the working size,
# is what decides a backlit room: on the real frame (face at luma 23) the torso
# came out at alpha 0.38 at 0.25, 0.47 at 0.5 and 0.86 at 1.0, for 16 / 25 / 67
# ms on the CPU session. The coarse pass is right for a lit room and cheap enough
# to run every frame; the fine pass is what recovers a body that the coarse pass
# drops entirely, so it is used only where the room is dark (see RatioChoice).
DOWNSAMPLE_RATIO = 0.25
DETAIL_RATIO = 1.0
# Frames of a given light before the choice changes, so a flickering exposure
# does not restart the recurrent state every other frame.
DARK_FRAMES = 4
LIGHT_FRAMES = 30


class RatioChoice:
    """Encoder detail for the current light, with hysteresis.

    Pure and model-free so the policy can be tested without a session: the
    switch itself has to reset RVM's recurrent state (its four state tensors are
    shaped by the ratio), which costs a couple of frames of soft alpha.
    """

    def __init__(self, dark_frames: int = DARK_FRAMES, light_frames: int = LIGHT_FRAMES) -> None:
        self.dark_frames = max(1, int(dark_frames))
        self.light_frames = max(1, int(light_frames))
        self.ratio = DOWNSAMPLE_RATIO
        self._dark = 0
        self._light = 0

    def update(self, dark: bool) -> float:
        """Feed one frame's verdict, get the ratio to use from here on."""
        if dark:
            self._light = 0
            self._dark += 1
            if self._dark >= self.dark_frames:
                self.ratio = DETAIL_RATIO
        else:
            self._dark = 0
            self._light += 1
            if self._light >= self.light_frames:
                self.ratio = DOWNSAMPLE_RATIO
        return self.ratio


class MatteModel:
    """RVM wrapper: one BGR frame in, one float32 alpha out."""

    def __init__(self, model_path: str | Path, providers: Optional[list[str]] = None,
                 size: tuple[int, int] = MATTE_SIZE) -> None:
        import onnxruntime as ort

        self.size = (int(size[0]), int(size[1]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.log_severity_level = 3
        # CPU on purpose. The model is small (15 MB): 8 ms at 512 px here, and
        # running it next to the DirectML swap/detector sessions in a separate
        # thread crashed onnxruntime on this driver. CPU also leaves the GPU for
        # the face, which is the part that needs it.
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers or ["CPUExecutionProvider"],
        )
        inputs = {item.name: item for item in self.session.get_inputs()}
        outputs = [item.name for item in self.session.get_outputs()]
        missing = [name for name in ("src", "r1i", "r2i", "r3i", "r4i", "downsample_ratio")
                   if name not in inputs]
        if missing:
            raise RuntimeError(f"matte model is missing inputs: {', '.join(missing)}")
        if "pha" not in outputs:
            raise RuntimeError("matte model has no alpha output ('pha')")
        self._state = self._empty_state()
        self._ratio = DOWNSAMPLE_RATIO

    def _empty_state(self) -> list[np.ndarray]:
        return [np.zeros((1, 1, 1, 1), dtype=np.float32) for _ in range(4)]

    def reset(self) -> None:
        self._state = self._empty_state()

    @property
    def provider(self) -> str:
        return self.session.get_providers()[0]

    def _use_ratio(self, ratio: Optional[float]) -> float:
        """Set the encoder detail, resetting the state when it changes.

        RVM's recurrent tensors are sized by the ratio, so a mid-stream change
        fails inside its Expand node unless the state is dropped first.
        """
        wanted = DOWNSAMPLE_RATIO if ratio is None else float(ratio)
        if wanted != self._ratio:
            self.reset()
            self._ratio = wanted
        return self._ratio

    def _run(self, frame_bgr: np.ndarray,
             ratio: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        ratio = self._use_ratio(ratio)
        height, width = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA)
        blob = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = blob.transpose(2, 0, 1)[None] / 255.0
        outputs = self.session.run(
            None,
            {
                "src": blob,
                "r1i": self._state[0],
                "r2i": self._state[1],
                "r3i": self._state[2],
                "r4i": self._state[3],
                "downsample_ratio": np.array([ratio], dtype=np.float32),
            },
        )
        by_name = dict(zip([item.name for item in self.session.get_outputs()], outputs))
        self._state = [by_name[name] for name in ("r1o", "r2o", "r3o", "r4o")]
        return by_name["pha"][0, 0], by_name["fgr"][0]

    def alpha(self, frame_bgr: np.ndarray, refine_alpha: bool = True,
              ratio: Optional[float] = None) -> np.ndarray:
        """Soft person alpha at the frame's own resolution, values 0..1."""
        alpha, _foreground = self._run(frame_bgr, ratio)
        if refine_alpha:
            # Trim the uncertain rim before upscaling: 1 ms at 512 px instead of
            # 6 ms at 720p, and the guided filter below keeps it soft.
            alpha = refine(alpha, feather=0.4)
        return guided_upsample(alpha, frame_bgr)

    def matte(self, frame_bgr: np.ndarray,
              ratio: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
        """Alpha plus the person's own colours under the soft rim.

        The second image is the frame with every semi-transparent pixel replaced
        by the colour of the person just inside the silhouette. The composite
        uses it only along the rim; it is what stops a bright camera wall from
        surviving as a grey fringe on dark hair.
        """
        alpha, _foreground = self._run(frame_bgr, ratio)
        alpha = guided_upsample(refine(alpha, feather=0.4), frame_bgr)
        return alpha, decontaminate(frame_bgr, alpha)


_SHARED: dict[tuple[str, tuple[str, ...]], "MatteModel"] = {}


def shared(model_path: str | Path, providers: Optional[list[str]] = None) -> "MatteModel":
    """One matte session per process.

    Creating a second DirectML session after the first has been released can
    crash onnxruntime on this driver. The app starts and stops the engine every
    time the user presses a button, so the session is kept alive and reused;
    ``reset()`` clears the recurrent state for a new stream.
    """
    key = (str(model_path), tuple(providers or ()))
    model = _SHARED.get(key)
    if model is None:
        model = MatteModel(model_path, providers)
        _SHARED[key] = model
    model.reset()
    return model


def refine(alpha: np.ndarray, threshold: float = 0.06, hardness: float = 1.25,
           feather: float = 0.6) -> np.ndarray:
    """Trim the low-confidence rim and keep the edge soft.

    ``alpha < threshold`` is camera background that the network was not sure
    about. Cutting it stops the bright halo; dividing by ``1 - threshold``
    rescales so a fully covered pixel stays at 1.0.
    """
    alpha = np.asarray(alpha, dtype=np.float32)
    soft = np.clip((alpha - threshold) / max(1e-3, 1.0 - threshold), 0.0, 1.0)
    soft = np.power(soft, hardness)
    if feather > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), feather)
    return soft


def guided_upsample(alpha: np.ndarray, frame_bgr: np.ndarray,
                    radius: int = 5, eps: float = 1e-4) -> np.ndarray:
    """Upsample a small matte along the full-resolution colour edges.

    Bilinear upscaling of a 512 px matte gives a soft, sometimes stair-stepped
    hairline. A guided filter uses the frame itself as the edge oracle: where
    the photo has a sharp colour boundary, the alpha snaps to it. Runs at half
    resolution (about 4 ms) and is upscaled once.
    """
    height, width = frame_bgr.shape[:2]
    gw = max(8, width // 2)
    gh = max(8, height // 2)
    guide = cv2.cvtColor(cv2.resize(frame_bgr, (gw, gh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    target = cv2.resize(alpha, (gw, gh), interpolation=cv2.INTER_AREA)
    window = (2 * radius + 1, 2 * radius + 1)
    mean_guide = cv2.blur(guide, window)
    mean_target = cv2.blur(target, window)
    var_guide = cv2.blur(guide * guide, window) - mean_guide * mean_guide
    cov = cv2.blur(guide * target, window) - mean_guide * mean_target
    a = cov / (var_guide + eps)
    b = mean_target - a * mean_guide
    out = cv2.blur(a, window) * guide + cv2.blur(b, window)
    return np.clip(cv2.resize(out, (width, height), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)


def decontaminate(frame_bgr: np.ndarray, alpha: np.ndarray,
                   scale: int = 4, radius: float = 3.0) -> np.ndarray:
    """Fill the soft rim with the person's colour from just inside it.

    A matting network cannot tell a half-covered hair strand from a half-covered
    wall pixel, so the rim it returns carries the camera background's colour.
    RVM's own ``fgr`` is no better: on a black hairline over a white wall it
    estimates a pale grey. Compositing that through a soft alpha leaves a halo
    that makes the person look pasted on, and no amount of alpha sharpening
    removes it because the colour itself is wrong.

    The fix is a colour-bleed correction: take the average of the confident
    person pixels nearby, weight by coverage, and use it for every pixel whose
    alpha is not fully opaque. At half coverage the blend is then half hair and
    half room, which is what a soft edge should look like. Runs at quarter
    resolution, about a millisecond.

    The blur radius is a *full-frame* reach, not a small-grid one: at quarter
    resolution the old radius of 9 pulled colour from 36 px away, which smeared
    the jaw and cheek edges into a smooth blob and cost real identity detail.
    """
    height, width = frame_bgr.shape[:2]
    # Every other pixel is enough to bound the person: the 2 px error this can
    # make is inside the pad below, and a full-frame scan costs 2.8 ms.
    ys, xs = np.nonzero(alpha[::2, ::2] > 0.02)
    if ys.size == 0:
        return frame_bgr
    # The person only occupies part of the frame. Working on their bounding box
    # instead of the whole 720p plane is most of the difference between 6 ms
    # and 2 ms a frame.
    pad = int(radius * scale) + 8
    y0 = max(0, int(ys.min()) * 2 - pad)
    y1 = min(height, int(ys.max()) * 2 + pad + 1)
    x0 = max(0, int(xs.min()) * 2 - pad)
    x1 = min(width, int(xs.max()) * 2 + pad + 1)
    crop = frame_bgr[y0:y1, x0:x1]
    crop_alpha = alpha[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]

    small = cv2.resize(crop, (max(2, cw // scale), max(2, ch // scale)),
                       interpolation=cv2.INTER_AREA)
    coverage = cv2.resize(crop_alpha, (small.shape[1], small.shape[0]),
                          interpolation=cv2.INTER_AREA)
    inside = (coverage > 0.85).astype(np.float32)
    window = (0, 0)
    num = cv2.GaussianBlur(small.astype(np.float32) * inside[:, :, None], window, radius)
    den = cv2.GaussianBlur(inside, window, radius)[:, :, None]
    colour = num / np.maximum(den, 1e-3)
    # An isolated sliver with no confident interior nearby keeps its own colour.
    empty = (den[:, :, 0] < 1e-3)[:, :, None]
    colour = np.where(empty, small.astype(np.float32), colour)
    colour = cv2.resize(np.clip(colour, 0, 255).astype(np.uint8), (cw, ch),
                        interpolation=cv2.INTER_LINEAR)
    # Only the rim needs the new colour: a fully transparent pixel is never
    # composited, and a fully opaque one is already pure person. A masked copy is
    # a SIMD blit; boolean-index assignment of the same 100k pixels cost 8 ms.
    band = cv2.bitwise_and(cv2.compare(crop_alpha, 0.02, cv2.CMP_GT),
                           cv2.compare(crop_alpha, 0.985, cv2.CMP_LT))
    out = frame_bgr.copy()
    cv2.copyTo(colour, band, out[y0:y1, x0:x1])
    return out


def background_estimate(frame_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """A blurry picture of what is behind the person, at quarter resolution.

    Computed from the pixels the matte is confident are background, weighted
    box-blurred so holes are filled from their neighbours. The person's own
    colours cannot leak into it.
    """
    height, width = frame_bgr.shape[:2]
    scale = 0.25
    small = cv2.resize(frame_bgr, (max(2, int(width * scale)), max(2, int(height * scale))),
                       interpolation=cv2.INTER_AREA)
    keep = (cv2.resize(alpha, (small.shape[1], small.shape[0]),
                       interpolation=cv2.INTER_AREA) < 0.2).astype(np.float32)
    window = (17, 17)
    num = cv2.blur(small.astype(np.float32) * keep[:, :, None], window)
    den = cv2.blur(keep, window)[:, :, None]
    background = num / np.maximum(den, 1e-3)
    # Where there is no background evidence at all, fall back to the frame mean.
    empty = (den[:, :, 0] < 1e-3)
    if empty.any():
        background[empty] = small.reshape(-1, 3).mean(axis=0)
        background = cv2.blur(background, (9, 9))
    return cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)


def unmix(frame_bgr: np.ndarray, alpha: np.ndarray,
          background: Optional[np.ndarray] = None) -> np.ndarray:
    """Recover the person's own colours from a soft matte.

    ``frame = alpha * foreground + (1 - alpha) * background``. Solving for the
    foreground is what removes the old room from the hair edge instead of just
    fading it, which is the difference between a cutout and a green-screen
    paste.
    """
    if float(alpha.min()) >= 0.999:
        return frame_bgr
    bg = background if background is not None else background_estimate(frame_bgr, alpha)
    a = np.clip(alpha, 0.22, 1.0)[:, :, None]
    foreground = (frame_bgr.astype(np.float32) - (1.0 - a) * bg.astype(np.float32)) / a
    # A very low alpha is mostly noise: fade those pixels back to the frame so
    # the unmix does not amplify clipping into white speckles. They contribute
    # almost nothing after compositing anyway.
    trust = np.clip((alpha - 0.05) / 0.35, 0.0, 1.0)[:, :, None]
    mixed = frame_bgr.astype(np.float32) * (1.0 - trust) + foreground * trust
    return np.clip(mixed, 0, 255).astype(np.uint8)
