# MimicCam 0.2.2 — the copyright in its own file, the name on everything

**MimicCam is © 2026 Ray Bekson (bigrayszn) — <https://github.com/bigrayszn/mimiccam>.**
Open source under **GPL-2.0-or-later**: read it, fork it, ship it, and keep the notice.
The copyright and the original authorship stay with Ray Bekson.

0.2.1 put the notice into the LICENSE file itself. That made the repository report its
licence as *unknown* — GitHub only declares "GPL-2.0" when the file starts with the
licence — which is the wrong trade for an open-source project. So the arrangement is
now the standard one:

- **[LICENSE](LICENSE)** is the verbatim GNU GPL v2. The repository badge reads
  **GPL-2.0** again.
- **[COPYRIGHT.md](COPYRIGHT.md)** is the statement of authorship: who owns the
  project, what the licence permits and requires, and a table of every place the
  notice appears.
- Nothing else changed in what the app says: `--version` still prints the copyright
  and the repository, the window header still shows it, both installers still name
  Ray Bekson as the publisher, and the GitHub repository description still leads with
  his name.

## Everything the notice touches

| artifact | what it says |
|---|---|
| [COPYRIGHT.md](COPYRIGHT.md) | the ownership statement, licence obligations, and where the notice appears |
| [LICENSE](LICENSE) | the verbatim GPL-2.0 text, so the repository declares **GPL-2.0** |
| [README.md](README.md) | first paragraph, above the download link |
| [INSTALL.md](INSTALL.md) | first two lines |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | the project copyright; components keep their own |
| `MimicCam.exe --version` | `MimicCam 0.2.2 - Copyright (C) 2026 Ray Bekson (bigrayszn) - GPL-2.0-or-later` plus the repository URL |
| the app window | `Copyright (C) 2026 Ray Bekson (bigrayszn)` in the header, repository and licence in its tooltip |
| `installer/*.iss` | publisher **Ray Bekson (bigrayszn)**; `AppUrl` is the repository |
| `mimiccam/__init__.py` | `__author__`, `__copyright__`, `__url__`, `__license__` — the single source the app reads |
| GitHub repository | description names the author; licence reads GPL-2.0 |

Four tests in `tests/test_core.py` pin it: the copyright strings, COPYRIGHT.md, the
verbatim LICENSE and the `--version` output.

## Unchanged

- The free engine's **garment transplant**, tracked to your torso: collar 18 px off on
  a 70 px lean against 141 px pinned to the face, 3.2 ms a frame.
- The **matte's second gear** in dark rooms: torso alpha 0.47 → 0.89 at 31.7 vs 32.0 fps.
- **205 unit tests** (`python -m pytest tests/ -q`).

## Install

Download `MimicCam-0.2.2-win64-portable.zip` and `SHASUMS256.txt` below, extract to a
**short** path (e.g. `C:\MimicCam` — Windows cannot load the virtual-camera DLL from a
very deep folder) and run `MimicCam.exe`. Windows 10 or 11, 64-bit; any DirectX 12 GPU
for the free engine, an NVIDIA GPU for Lucy mode; OBS Studio for its virtual-camera
driver. Full instructions: [INSTALL.md](INSTALL.md).

## Licensing

GPL-2.0-or-later, © 2026 Ray Bekson (bigrayszn). Model licences: YuNet MIT, InsightFace
research-only, RVM GPL-3.0, LaMa Apache-2.0, GFPGAN Apache-2.0 — see
`THIRD_PARTY_NOTICES.md`. Lucy 2.5 is a Decart model; not affiliated with Decart.
