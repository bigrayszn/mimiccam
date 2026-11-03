# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""NVIDIA GPU detection and adapter selection.

MimicCam is NVIDIA-only by design: the pipeline assumes a working NVIDIA driver
and uses it for pixel conversion, hardware decode of the remote stream and the
preview path. This module reports what is actually present instead of guessing.

It also answers the other half of that question: *which* DirectX adapter the
inference sessions land on. DirectML's ``device_id`` is a DXGI adapter index, so
the only honest way to pin it to the NVIDIA card is to enumerate the adapters in
DXGI order and hand the NVIDIA one's index to every session. Left to itself,
DirectML takes adapter 0, which is the discrete GPU here but is not guaranteed to
be on a hybrid laptop after a driver update.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import threading
import uuid
from typing import Optional

NVIDIA_VENDOR_ID = 0x10DE

# IID_IDXGIFactory2. The older IIDs are not exposed by every dxgi.dll build, and
# IDXGIFactory2 still answers EnumAdapters from its IDXGIFactory half.
_IID_IDXGIFACTORY2 = "50c83a1c-e072-4c48-87b0-3630fa36a6d0"

_QUERY = "name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu"


def _run(cmd: list[str], timeout: float = 8.0) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _nvidia_smi() -> Optional[str]:
    if shutil.which("nvidia-smi"):
        return "nvidia-smi"
    for candidate in (
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        r"C:\Windows\System32\nvidia-smi.exe",
    ):
        from pathlib import Path

        if Path(candidate).exists():
            return candidate
    return None


def nvidia_info() -> Optional[dict]:
    """Return GPU facts, or None when no NVIDIA driver is available."""
    smi = _nvidia_smi()
    if not smi:
        return None
    out = _run([smi, f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 6:
        return None
    name, driver, total, used, util, temp = parts[:6]

    info = {
        "name": name,
        "driver": driver,
        "vram_total_mb": _as_int(total),
        "vram_used_mb": _as_int(used),
        "utilization_pct": _as_int(util),
        "temperature_c": _as_int(temp),
        "nvenc": False,
    }

    # NVENC session counters only exist on cards that have an encoder block.
    enc = _run([smi, "--query-gpu=encoder.stats.sessionCount", "--format=csv,noheader,nounits"])
    if enc and enc.splitlines()[0].strip() not in ("", "N/A", "[N/A]"):
        info["nvenc"] = True
        info["nvenc_sessions"] = _as_int(enc.splitlines()[0])
    return info


def _as_int(value: str) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def require_nvidia(info: Optional[dict]) -> Optional[str]:
    """Why the AI engine must not run here, or None when the GPU is fine.

    MimicCam is NVIDIA-only on purpose: the pipeline leans on the driver for pixel
    conversion and remote-stream decode, and a machine without it produces a
    slideshow with a swapped face. Refusing up front beats pretending.
    """
    if info is None:
        return (
            "No NVIDIA GPU or driver was found.\n\n"
            "MimicCam's AI engine requires an NVIDIA GPU with a current driver. "
            "Install or repair the driver from nvidia.com, then restart MimicCam.\n\n"
            "Passthrough mode still works, if you only want to test the camera path."
        )
    return None


def describe(info: Optional[dict]) -> str:
    if not info:
        return "No NVIDIA GPU detected"
    name = info.get("name", "NVIDIA GPU")
    vram = info.get("vram_total_mb")
    driver = info.get("driver", "?")
    suffix = f", {vram / 1024:.0f} GB VRAM" if vram else ""
    return f"{name}{suffix} - driver {driver}"


# -- DirectML adapter selection --------------------------------------------
class _DXGI_ADAPTER_DESC(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", ctypes.c_byte * 8),
    ]


def dxgi_adapters() -> list[dict]:
    """Every DirectX adapter, in enumeration order.

    The list index is the ``device_id`` DirectML wants. Anything unexpected
    (no dxgi.dll, a moved COM vtable) yields an empty list rather than an
    exception: this runs at startup, where a wrong guess must not kill the app.
    """
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
        create = dxgi.CreateDXGIFactory
        create.restype = ctypes.c_long
        create.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        iid = ctypes.create_string_buffer(uuid.UUID(_IID_IDXGIFACTORY2).bytes_le, 16)
        factory = ctypes.c_void_p()
        if create(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory:
            return []

        vtable = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        # IDXGIFactory: IUnknown(3) + IDXGIObject(4) methods, then EnumAdapters.
        enum_adapters = ctypes.WINFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        )(vtable[7])
    except Exception:
        return []

    adapters: list[dict] = []
    for index in range(16):
        adapter = ctypes.c_void_p()
        if enum_adapters(factory, index, ctypes.byref(adapter)) != 0 or not adapter:
            break
        try:
            avtable = ctypes.cast(
                adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
            ).contents
            # IDXGIAdapter: EnumOutputs, GetDesc, CheckInterfaceSupport.
            get_desc = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_DXGI_ADAPTER_DESC)
            )(avtable[8])
            desc = _DXGI_ADAPTER_DESC()
            if get_desc(adapter, ctypes.byref(desc)) != 0:
                continue
        except Exception:
            continue
        adapters.append(
            {
                "index": index,
                "name": desc.Description,
                "vendor": desc.VendorId,
                "vram_mb": desc.DedicatedVideoMemory // (1024 * 1024),
            }
        )
    return adapters


_ADAPTER_LOCK = threading.Lock()
_ADAPTER: Optional[dict] = None
_ADAPTER_LOOKED = False


def nvidia_adapter(refresh: bool = False) -> Optional[dict]:
    """The first DirectX adapter made by NVIDIA, or None.

    Cached: every model session asks for it, and enumerating DXGI is not free.
    """
    global _ADAPTER, _ADAPTER_LOOKED
    with _ADAPTER_LOCK:
        if refresh or not _ADAPTER_LOOKED:
            _ADAPTER_LOOKED = True
            _ADAPTER = next(
                (a for a in dxgi_adapters() if a["vendor"] == NVIDIA_VENDOR_ID), None
            )
        return _ADAPTER


def directml_providers(strict: bool = False) -> list:
    """onnxruntime providers pinned to the NVIDIA adapter.

    ``strict`` turns a missing NVIDIA card into a failure instead of a quiet
    CPU fallback: the free engine's whole promise is 30 fps, and a CPU-only
    session would deliver a slideshow the user never agreed to.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        if strict:
            raise RuntimeError("onnxruntime is not installed")
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    if "DmlExecutionProvider" not in available:
        if strict:
            raise RuntimeError(
                "This build has no DirectML provider, so it cannot run the models on "
                "the NVIDIA GPU. Install onnxruntime-directml."
            )
        return ["CPUExecutionProvider"]

    adapter = nvidia_adapter()
    if adapter is None:
        if strict:
            raise RuntimeError(
                "No NVIDIA DirectX adapter was found. MimicCam runs its models on the "
                "NVIDIA GPU only; it will not fall back to the CPU or to integrated "
                "graphics.\n\nInstall or repair the NVIDIA driver, then restart MimicCam."
            )
        return ["CPUExecutionProvider"]

    return [
        ("DmlExecutionProvider", {"device_id": adapter["index"]}),
        "CPUExecutionProvider",
    ]


def adapter_name(providers: Optional[list] = None) -> str:
    """Human-readable adapter for a provider list built by directml_providers."""
    pinned = None
    for entry in providers or []:
        if isinstance(entry, tuple) and entry[0] == "DmlExecutionProvider":
            pinned = entry[1].get("device_id")
    if pinned is None:
        return "CPU"
    for adapter in dxgi_adapters():
        if adapter["index"] == pinned:
            return adapter["name"]
    return f"DirectX adapter {pinned}"
