# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""XSeg occlusion: stop a hand, hair or a microphone being painted over.

The swap network renders a whole face from the character photo; it has no idea
that a hand is in front of the mouth. Pasted at full strength, the render
replaces the hand with the character's cheek - the smeared, melting look that
gives a swap away faster than a soft jaw.

DeepFaceLab's XSeg model answers exactly one question per pixel: *is this face
skin, or is something in front of it?* It was trained on hand-labelled frames
that deliberately include hands, hair, glasses and phone mics, which is why one
256 px pass is enough to keep them. This module runs that pass and turns it into
the one thing the paste needs: a blend weight that is 1 on skin and 0 on the
thing covering it.

The graph takes ``NHWC`` input in 0..1 BGR - DeepFaceLab's own layout, not the
``NCHW`` the swap networks use - and returns a single-channel skin mask. Both
were read off the graph at load time rather than assumed.
"""

from __future__ import annotations

import numpy as np

# XSeg's training resolution. The model is fully convolutional, but its masks
# were tuned at 256 and the extra detail above that is not worth the latency.
SIZE = 256

# The model's sigmoid is confident in its interior and vague at the boundary
# (measured on a real crop: 1.00 across the cheeks, 0.00 on a hand, most of the
# ambiguity inside this band). Anything below LOW is treated as an occluder and
# anything above HIGH as skin, with a straight ramp between, so the mask edge
# fades instead of cutting a hard line through a jaw.
LOW = 0.30
HIGH = 0.80

# Softens the returned weight. A hand's edge is a few pixels wide at webcam
# scale; blurring by that much is what stops the render showing a rectangular
# bite taken out of the face where the mask stepped.
FEATHER = 5


def _session(model_path, providers=None):
    """An onnxruntime session, kept private so this module stays dependency-light."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=options, providers=providers or ["CPUExecutionProvider"])


class Occluder:
    """Turns an aligned face crop into "swap here / keep the frame here" weights."""

    def __init__(self, model_path, providers=None) -> None:
        self.model_path = str(model_path)
        self.session = _session(self.model_path, providers)
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        # NHWC or NCHW, decided from the graph rather than from the generation of
        # XSeg that happens to be installed.
        shape = list(inputs[0].shape)
        self.nhwc = len(shape) == 4 and int(shape[3] or 0) == 3
        self.size = SIZE
        self._open_runs = 0

    def skin(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Skin probability for one aligned crop: (size, size) float32 in 0..1."""
        size = int(crop_bgr.shape[0])
        work = crop_bgr
        if size != self.size:
            import cv2

            work = cv2.resize(crop_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        blob = (work.astype(np.float32) / 255.0)
        blob = blob[None] if self.nhwc else blob.transpose(2, 0, 1)[None]
        blob = np.ascontiguousarray(blob)
        output = self.session.run(None, {self.input_name: blob})[0]
        mask = np.squeeze(output).astype(np.float32)
        if mask.ndim != 2:
            raise RuntimeError(f"unexpected XSeg output shape {output.shape}")
        return np.clip(mask, 0.0, 1.0)

    def protection(self, crop_bgr: np.ndarray, feather: int = FEATHER) -> np.ndarray:
        """Blend weight for the paste: 1 on face skin, 0 on whatever covers it.

        Deliberately returns the *occlusion* reading only. Combining it with the
        face-shaped mask belongs to the swapper, which knows the paste geometry.
        """
        import cv2

        mask = self.skin(crop_bgr)
        mask = np.clip((mask - LOW) / max(1e-6, HIGH - LOW), 0.0, 1.0)
        if feather and feather > 0:
            odd = int(feather) | 1
            mask = cv2.GaussianBlur(mask, (odd, odd), 0)
        return mask.astype(np.float32)
