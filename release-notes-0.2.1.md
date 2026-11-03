# MimicCam 0.2.1 — Ray Bekson's project, stated everywhere it matters

**MimicCam is © 2026 Ray Bekson (bigrayszn) — <https://github.com/bigrayszn/mimiccam>.**
Open source under **GPL-2.0-or-later**: read it, fork it, ship it, and keep the notice.
The copyright and the original authorship stay with Ray Bekson.

0.2.0 shipped the character's clothes and the dark-room matte. This release makes one
thing unambiguous in every artifact a user can reach: whose project this is.

## Where the ownership now appears

| artifact | what it says |
|---|---|
| [LICENSE](LICENSE) | opens with `Copyright (C) 2026 Ray Bekson (bigrayszn)` and the repository URL, before the GPL text, and names the third-party components as keeping their own licences |
| `MimicCam.exe --version` (new) | `MimicCam 0.2.1 - Copyright (C) 2026 Ray Bekson (bigrayszn) - GPL-2.0-or-later` and the repository URL |
| the app window | a discreet `Copyright (C) 2026 Ray Bekson (bigrayszn)` in the header, with the licence and repo in its tooltip |
| README, top of page | the same statement, above the download link |
| INSTALL.md | first two lines, before any instructions |
| THIRD_PARTY_NOTICES.md | the project copyright, and explicitly *not* claimed over the components' licences |
| `mimiccam/__init__.py` | `__author__`, `__copyright__`, `__url__`, `__license__` — the single source the app and `--version` both read |
| both installers | publisher is **Ray Bekson (bigrayszn)**, and `AppUrl` points at the repository |
| this release and the GitHub repository description | name the author in the first line |

`tests/test_core.py` now pins it: the copyright strings, the `LICENSE` header and the
`--version` output are all asserted, so a later edit cannot quietly drop the name.

## Unchanged from 0.2.0

- The free engine's **garment transplant**, fitted to your torso and tracked with your
  body: the collar lands 18 px off on a 70 px lean, against 141 px when pinned to the
  face, for 3.2 ms a frame.
- The **matte's second gear** in dark rooms: RVM at `downsample_ratio` 1.0 instead of
  0.25 takes the wardrobe's torso alpha from 0.47 to 0.89 at 31.7 vs 32.0 fps.
- 204 unit tests (`python -m pytest tests/ -q`).

## Install

Download `MimicCam-0.2.1-win64-portable.zip` and `SHASUMS256.txt` below, extract to a
**short** path (e.g. `C:\MimicCam` — Windows cannot load the virtual-camera DLL from a
very deep folder) and run `MimicCam.exe`. Windows 10 or 11, 64-bit; any DirectX 12 GPU
for the free engine, an NVIDIA GPU for Lucy mode; OBS Studio for its virtual-camera
driver. Full instructions: [INSTALL.md](INSTALL.md).

The installer is built from this repository — `installer/public.iss` (app only) or
`installer/setup.iss` (single file with models) — with Inno Setup 6.

## Licensing

GPL-2.0-or-later, © 2026 Ray Bekson (bigrayszn). Model licences: YuNet MIT, InsightFace
research-only, RVM GPL-3.0, LaMa Apache-2.0, GFPGAN Apache-2.0 — see
`THIRD_PARTY_NOTICES.md`. Lucy 2.5 is a Decart model; this project is not affiliated
with Decart.
