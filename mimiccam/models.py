# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""On-device model files for the free, local engine.

Nothing here talks to a paid API. The local engine needs three open-weight
files, downloaded once (~640 MB total) into ``%APPDATA%\\MimicCam\\models``.
Studio background adds two more, fetched only when that mode is enabled:

=====================  ======================================  ============
file                   job                                      size
=====================  ======================================  ============
``yunet.onnx``         face detection + 5 landmarks (OpenCV Zoo) 0.2 MB
``w600k_r50.onnx``     ArcFace identity embedding                166 MB
``inswapper_128.onnx`` 128x128 identity swap                     528 MB
``hyperswap_1a_256.onnx`` 256x256 identity swap (the tier above)  402 MB
``xseg_3.onnx``        occlusion mask for that tier               70 MB
``rvm_mobilenetv3_fp32.onnx`` person matte, temporally stable    15 MB
``lama_fp32.onnx``     large-mask inpainting for the room plate  208 MB
=====================  ======================================  ============

They are deliberately *not* bundled in the .exe: they are large, they are
updated independently of the app, and the InsightFace weights carry a
research-only licence that the user has to accept knowingly. The download is a
single HTTP GET per file, resumable by re-running, and every file is size
checked before it is used.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

CHUNK = 1 << 20
ENV_MODELS_DIR = "MIMICCAM_MODELS_DIR"
# Built locally from the fp16 download, never fetched: see convert_swapper_fp32.
SWAPPER_FP32_FILENAME = "hyperswap_1a_256_fp32.onnx"
_UA = "MimicCam/0.1 (+https://github.com/your-org/mimiccam)"

INSIGHTFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7"

# The identity and swap models are always needed. The studio models are only
# downloaded when the user turns the reference-room background on.
#
# The 256 px tier *is* the swap network now, so it is core along with its XSeg
# mask. inswapper_128 stays downloadable - a rig that cannot run the 256 px
# network still swaps on it - but a fresh install no longer spends 528 MB on a
# network it will not use.
CORE_KEYS = ("detector", "identity", "swapper_256", "occluder")
FALLBACK_KEYS = ("swapper",)
# The 256 px tier: the swap network that replaced inswapper_128 and the XSeg
# mask that keeps hands and hair out of the paste. Optional - a rig holding only
# the core files keeps working on the 128 px network. The fp32 copy of the swap
# network is derived from swapper_256 (convert_swapper_fp32), not downloaded.
TIER_KEYS = ("swapper_256", "occluder")
STUDIO_KEYS = ("matte", "inpaint")
# Optional: only fetched when the user leaves the Detail pass on.
DETAIL_KEYS = ("enhancer",)


class ModelError(RuntimeError):
    """A model file is missing, truncated or could not be fetched."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str
    url: str
    size_bytes: int
    purpose: str
    licence: str
    archive_member: str = ""

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="detector",
        filename="yunet.onnx",
        url="https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        size_bytes=232589,
        purpose="Face detection and 5-point landmarks",
        licence="MIT (OpenCV Zoo)",
    ),
    ModelSpec(
        key="identity",
        filename="w600k_r50.onnx",
        url=f"{INSIGHTFACE_URL}/buffalo_l.zip",
        size_bytes=288621354,
        purpose="ArcFace identity embedding of your character photo",
        licence="InsightFace: non-commercial research use only",
        archive_member="w600k_r50.onnx",
    ),
    ModelSpec(
        key="swapper",
        filename="inswapper_128.onnx",
        url=f"{INSIGHTFACE_URL}/inswapper_128.onnx",
        size_bytes=554253681,
        purpose="128x128 identity swap network",
        licence="InsightFace: non-commercial research use only",
    ),
    ModelSpec(
        key="swapper_256",
        filename="hyperswap_1a_256.onnx",
        url="https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1a_256.onnx",
        size_bytes=402742682,
        purpose="256x256 identity swap network (HyperSwap 1a): 4x the pixels, own face mask",
        licence="ResearchRAIL (FaceFusion Labs): non-commercial research use only",
    ),
    ModelSpec(
        key="occluder",
        filename="xseg_3.onnx",
        url="https://huggingface.co/facefusion/models-3.2.0/resolve/main/xseg_3.onnx",
        size_bytes=70327709,
        purpose="XSeg occlusion mask: keeps hands, hair and objects in front of the face",
        licence="GPL-3.0 (DeepFaceLab); compatible with this app's GPL-2.0-or-later",
    ),
    ModelSpec(
        key="matte",
        filename="rvm_mobilenetv3_fp32.onnx",
        url="https://huggingface.co/eafish/web-onnx/resolve/"
        "576473dca1b5792d8c30753b21325edc4063440b/rvm_mobilenetv3_fp32.onnx",
        size_bytes=14975696,
        purpose="Person matte for the studio background (Robust Video Matting)",
        licence="GPL-3.0 (Robust Video Matting); compatible with this app's GPL-2.0-or-later",
    ),
    ModelSpec(
        key="inpaint",
        filename="lama_fp32.onnx",
        url="https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
        size_bytes=208044816,
        purpose="Removes the reference person when building the studio room",
        licence="Apache-2.0 (LaMa); trained on Places2 (CC-BY 4.0)",
    ),
    ModelSpec(
        key="enhancer",
        filename="gfpgan_1.4.onnx",
        url="https://huggingface.co/Neus/GFPGANv1.4/resolve/main/GFPGANv1.4.onnx",
        size_bytes=340256686,
        purpose="Restores detail in the 128 px swapped face (GFPGAN v1.4)",
        # Apache-2.0 is the grant on Tencent's release; FFHQ, which the weights
        # were trained on, is CC BY-NC-SA. See FACE_ENHANCERS.md for the survey.
        licence="Apache-2.0 (GFPGAN, TencentARC); trained on FFHQ (CC BY-NC-SA 4.0)",
    ),
)

BY_KEY = {spec.key: spec for spec in SPECS}
TOTAL_BYTES = sum(spec.size_bytes for spec in SPECS)

Progress = Callable[[str, int, int], None]


def models_dir() -> Path:
    """Where model files live. Overridable so tests never touch real ones."""
    override = os.environ.get(ENV_MODELS_DIR, "").strip()
    if override:
        path = Path(override)
    else:
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / ".config"
        path = root / "MimicCam" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def path(key: str) -> Path:
    return models_dir() / BY_KEY[key].filename


def swapper_fp32_path() -> Path:
    """The fp32 copy of the 256 px network, built from the fp16 download."""
    return models_dir() / SWAPPER_FP32_FILENAME


def _fp32_complete() -> bool:
    """True when the fp32 copy exists and looks like a full rewrite."""
    try:
        size = swapper_fp32_path().stat().st_size
        source = path("swapper_256").stat().st_size
    except OSError:
        return False
    # fp16 -> fp32 doubles every weight, so a stub or half-written file is not it.
    return size > source * 1.5


def swapper_path() -> Path:
    """The best swap network on disk: the 256 px tier when it is installed.

    Deliberately not an error when the tier is missing. A user who has the core
    three files gets the 128 px network and everything else exactly as before;
    the upgrade is additive.

    The fp32 copy wins over the fp16 download it was built from. DirectML runs
    the fp16 graph at half precision and loses the identity - on a real frame
    the pasted face scored 0.49 cosine against the character photo, where the
    same weights in fp32 scored 0.70 at 32 ms a render. See
    ``convert_swapper_fp32``.
    """
    if _fp32_complete():
        return swapper_fp32_path()
    if present("swapper_256"):
        return path("swapper_256")
    return path("swapper")


def swapper_precision() -> str:
    """``'fp32'``, ``'fp16'`` or ``''`` for the network ``swapper_path`` returns."""
    chosen = swapper_path()
    if chosen.name == SWAPPER_FP32_FILENAME:
        return "fp32"
    if chosen.name == BY_KEY["swapper_256"].filename:
        return "fp16"
    return ""


def tier_ready() -> bool:
    return present("swapper_256") and present("occluder")


def _looks_complete(spec: ModelSpec, target: Path) -> bool:
    try:
        size = target.stat().st_size
    except OSError:
        return False
    if size <= 0:
        return False
    # The archive member size is only known after extraction, so only the
    # direct downloads get an exact check.
    if spec.archive_member:
        return size > 100 * 1024 * 1024
    # Allow a small slack: mirrors occasionally differ by a few padding bytes.
    return abs(size - spec.size_bytes) <= max(4096, spec.size_bytes // 1000)


def present(key: str) -> bool:
    spec = BY_KEY[key]
    return _looks_complete(spec, models_dir() / spec.filename)


def selected(keys: Iterable[str]) -> list[ModelSpec]:
    """Specs for ``keys``. Pass ``None`` for every model, studio included."""
    if keys is None:
        return list(SPECS)
    return [BY_KEY[key] for key in keys]


def missing(keys: Iterable[str] = CORE_KEYS) -> list[ModelSpec]:
    return [spec for spec in selected(keys) if not present(spec.key)]


def is_ready(keys: Iterable[str] = CORE_KEYS) -> bool:
    return not missing(keys)


def total_missing_bytes(keys: Iterable[str] = CORE_KEYS) -> int:
    return sum(spec.size_bytes for spec in missing(keys))


# -- download -------------------------------------------------------------
def _fetch(url: str, target: Path, total_hint: int, on_chunk, cancel=None) -> int:
    """Stream one URL to ``target``; returns bytes written."""
    import urllib.request

    written = 0
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(request, timeout=30) as response:
        total = int(response.headers.get("Content-Length") or total_hint)
        with open(target, "wb") as handle:
            while True:
                if cancel is not None and cancel.is_set():
                    raise ModelError("Download cancelled")
                block = response.read(CHUNK)
                if not block:
                    break
                handle.write(block)
                written += len(block)
                on_chunk(written, total or total_hint)
    return written


def download(key: str, progress: Optional[Progress] = None, cancel=None) -> Path:
    """Fetch one model. Safe to call again: complete files are left alone."""
    spec = BY_KEY[key]
    target = path(key)
    if present(key):
        return target

    def report(done: int, total: int) -> None:
        if progress is not None:
            progress(key, done, total)

    partial = target.with_suffix(target.suffix + ".part")
    try:
        if spec.archive_member:
            archive = Path(tempfile.mkdtemp(prefix="mimiccam-dl-")) / "buffalo_l.zip"
            try:
                _fetch(spec.url, archive, spec.size_bytes, report, cancel)
                with zipfile.ZipFile(archive) as bundle:
                    with bundle.open(spec.archive_member) as source:
                        with open(partial, "wb") as out:
                            shutil.copyfileobj(source, out, CHUNK)
            finally:
                shutil.rmtree(archive.parent, ignore_errors=True)
        else:
            _fetch(spec.url, partial, spec.size_bytes, report, cancel)

        if not _looks_complete(spec, partial):
            raise ModelError(
                f"{spec.filename} downloaded incomplete "
                f"({partial.stat().st_size} bytes). Try again."
            )
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return target


def ensure(progress: Optional[Progress] = None, cancel=None,
           keys: Iterable[str] = CORE_KEYS) -> list[Path]:
    """Download everything that is missing. Returns the paths in use."""
    chosen = selected(keys)
    for spec in [s for s in chosen if not present(s.key)]:
        download(spec.key, progress=progress, cancel=cancel)
    if any(spec.key == "swapper_256" for spec in chosen):
        prepare_swapper_fp32(progress=progress, cancel=cancel)
    return [path(spec.key) for spec in chosen]


# -- the fp32 copy of the 256 px network ----------------------------------
def convert_swapper_fp32(progress: Optional[Progress] = None, cancel=None) -> Path:
    """Rewrite the fp16 256 px network as fp32, next to it. No download.

    FaceFusion ships HyperSwap in fp16, and DirectML runs an fp16 graph at half
    precision - which costs identity, not just sharpness: on the frame this
    tier was built against, the pasted face scored **0.49** cosine against the
    character photo from the fp16 file and **0.70** from this fp32 copy (19 and
    32 ms a render; the 128 px network it replaces scores 0.87 at 37 ms, and a
    fused fp16 graph scores 0.16, so fp16 is not usable on this GPU at all).

    The weights are identical and no file is fetched: every float16 tensor in
    the graph is re-typed float32. ~40 s once, 768 MB on disk.
    """
    source = path("swapper_256")
    if not present("swapper_256"):
        raise ModelError("the 256 px network is not installed yet")
    target = swapper_fp32_path()
    if _fp32_complete():
        return target
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, numpy_helper
    except ImportError as exc:  # pragma: no cover - onnx is a hard dependency
        raise ModelError(
            "The 'onnx' package is needed to prepare the 256 px network. "
            "Install it with: pip install onnx"
        ) from exc

    total = source.stat().st_size

    def report(share: float) -> None:
        if progress is not None:
            progress("swapper_256", int(share * total), total)

    graph = onnx.load(str(source))
    initialisers = graph.graph.initializer
    for index, init in enumerate(initialisers):
        if cancel is not None and cancel.is_set():
            raise ModelError("Conversion cancelled")
        if init.data_type == TensorProto.FLOAT16:
            array = numpy_helper.to_array(init).astype(np.float32)
            init.CopyFrom(numpy_helper.from_array(array, init.name))
        if index % 16 == 0:
            report(0.9 * index / max(1, len(initialisers)))

    def retype(sequence) -> None:
        for value in sequence:
            if value.type.tensor_type.elem_type == TensorProto.FLOAT16:
                value.type.tensor_type.elem_type = TensorProto.FLOAT

    retype(graph.graph.input)
    retype(graph.graph.output)
    retype(graph.graph.value_info)
    for node in graph.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    attr.t.CopyFrom(numpy_helper.from_array(
                        numpy_helper.to_array(attr.t).astype(np.float32)))

    onnx.checker.check_model(graph)
    report(0.95)
    partial = target.with_name(target.name + ".part")
    try:
        onnx.save(graph, str(partial))
        partial.replace(target)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    report(1.0)
    return target


def prepare_swapper_fp32(progress: Optional[Progress] = None,
                         cancel=None) -> Optional[Path]:
    """Build the fp32 copy if the tier is installed and it is not there yet.

    Best effort by design. The fp16 network still swaps, just with a weaker
    likeness, so a full disk - or a machine where the rewrite fails for any
    other reason - must not turn a download that otherwise succeeded into an
    error the user cannot get past.
    """
    if _fp32_complete():
        return swapper_fp32_path()
    if not present("swapper_256"):
        return None
    try:
        return convert_swapper_fp32(progress=progress, cancel=cancel)
    except (OSError, MemoryError, ModelError):
        return None


def licence_summary() -> str:
    lines = ["Models downloaded by MimicCam:"]
    for spec in SPECS:
        lines.append(f"  {spec.filename} - {spec.purpose} - {spec.licence}")
    lines.append("")
    lines.append(
        "The InsightFace weights are licensed for non-commercial research. "
        "Use them for personal projects; get a commercial licence from "
        "InsightFace before shipping a paid product."
    )
    return "\n".join(lines)


def describe_missing(specs: Iterable[ModelSpec]) -> str:
    specs = list(specs)
    if not specs:
        return "All local models are installed."
    total = sum(spec.size_bytes for spec in specs)
    names = ", ".join(spec.filename for spec in specs)
    return f"Missing {len(specs)} model file(s) ({total / 1048576:.0f} MB): {names}"
