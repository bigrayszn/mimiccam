# MimicCam 0.2.5 — the hardware is written down, and the copyright travels with it

**MimicCam is © 2026 Ray Bekson (bigrayszn) — <https://github.com/bigrayszn/mimiccam>.**
Open source under **GPL-2.0-or-later**; the name and the branding are reserved — see
[TRADEMARK.md](TRADEMARK.md).

0.2.5 is a release-packaging update. It records the machine the local engine was
measured on, the hardware it needs, and the copyright and licence in every place the
download can be read.

## Hardware this build was measured on

| part | value |
|---|---|
| OS | Windows 11 Pro, 64-bit |
| CPU | Intel Core i9-14900HX (24 cores, 32 threads) |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM |
| GPU driver | NVIDIA driver 610.47 |
| Integrated GPU | Intel UHD Graphics (present; not used by the local engine when the NVIDIA GPU is available) |

**Minimum for the local engine:** Windows 10 1809+ or Windows 11 (64-bit), any DirectX 12
GPU with DirectML, 16 GB RAM recommended, ~400 MB for the app plus ~908 MB for the local
models.

**Lucy 2.5 mode:** requires an NVIDIA GPU with a current driver. It refuses to start
otherwise.

Full details are in [INSTALL.md](INSTALL.md).

## Verified

- The version string reads 0.2.5 in the package, the installer scripts and the docs.
- Build and checksums are produced by `build.py` and the release workflow; see the
  checksums in `SHASUMS256.txt` attached to this release.

## Install

Download `MimicCam-0.2.5-win64-portable.zip` and `SHASUMS256.txt` below, extract to a
**short** path (e.g. `C:\MimicCam`), run `MimicCam.exe`. Windows 10/11 64-bit; any
DirectX 12 GPU for the local engine, an NVIDIA GPU for Lucy mode; OBS Studio for its
virtual-camera driver. Details: [INSTALL.md](INSTALL.md).

## Licensing

GPL-2.0-or-later, © 2026 Ray Bekson (bigrayszn). The name and branding are governed by
`TRADEMARK.md` and are not granted by the licence. Model licences: YuNet MIT,
InsightFace research-only, RVM GPL-3.0, LaMa Apache-2.0, GFPGAN Apache-2.0. Lucy 2.5 is
a Decart model; not affiliated with Decart.
