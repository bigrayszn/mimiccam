"""On-device model files for the free, local engine.

Nothing here talks to a paid API. The local engine needs three open-weight
files, downloaded once (~700 MB total) into ``%APPDATA%\\MimicCam\\models``.
Studio background adds two more, fetched only when that mode is enabled:

=====================  ======================================  ============
file                   job                                      size
=====================  ======================================  ============
``yunet.onnx``         face detection + 5 landmarks (OpenCV Zoo) 0.2 MB
``w600k_r50.onnx``     ArcFace identity embedding                166 MB
``inswapper_128.onnx`` 128x128 identity swap                     528 MB
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
_UA = "MimicCam/0.1 (+https://github.com/your-org/mimiccam)"

INSIGHTFACE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7"

# The identity and swap models are always needed. The studio models are only
# downloaded when the user turns the reference-room background on.
CORE_KEYS = ("detector", "identity", "swapper")
STUDIO_KEYS = ("matte", "inpaint")


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
    return [path(spec.key) for spec in chosen]


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
