# MimicCam 0.2.4 — the download carries its own paperwork

**MimicCam is © 2026 Ray Bekson (bigrayszn) — <https://github.com/bigrayszn/mimiccam>.**
Open source under **GPL-2.0-or-later**; the name and the branding are reserved — see
[TRADEMARK.md](TRADEMARK.md).

0.2.3 made the project's ownership explicit everywhere it can be read on the internet.
This release fixes the one place it was still missing: **the portable zip**.

The installer has shipped `LICENSE`, `NOTICE` and the copyright statement since 0.2.3,
but `MimicCam-<version>-win64-portable.zip` — the download most people take — carried
the application folder only. Anyone handed that zip, or finding it on a machine years
later, got no statement of who owns it.

Now `build.py` copies the paperwork into the application folder before zipping, so the
portable download contains, next to `MimicCam.exe`:

| file | what it says |
|---|---|
| `LICENSE` | the verbatim GNU GPL v2 |
| `NOTICE` | what every redistribution has to keep, and that the name is not licensed |
| `COPYRIGHT.md` | who owns the project, and where the notice appears |
| `TRADEMARK.md` | the name, the icon and the author's identity |
| `THIRD_PARTY_NOTICES.md` | the bundled models' own licences, which are not claimed |

Nothing else changed: the exe still reports `Company Name: Ray Bekson (bigrayszn)` and
`Legal Copyright: Copyright (C) 2026 Ray Bekson (bigrayszn) - GPL-2.0-or-later` under
Properties → Details, `--version` still prints the copyright and the repository, and
signing is still wired in and opt-in (`SIGNING.md`).

## Verified

- **211 unit tests** pass; the frozen exe builds, smoke-tests and signs as before.
- The released zip was downloaded and inspected: the five documents sit next to
  `MimicCam.exe`, and the exe's version resource carries the author.

## Install

Download `MimicCam-0.2.4-win64-portable.zip` and `SHASUMS256.txt` below, extract to a
**short** path (e.g. `C:\MimicCam`), run `MimicCam.exe`. Windows 10/11 64-bit; any
DirectX 12 GPU for the free engine, an NVIDIA GPU for Lucy mode; OBS Studio for its
virtual-camera driver. Details: [INSTALL.md](INSTALL.md).

## Licensing

GPL-2.0-or-later, © 2026 Ray Bekson (bigrayszn). The name and branding are governed by
`TRADEMARK.md` and are not granted by the licence. Model licences: YuNet MIT,
InsightFace research-only, RVM GPL-3.0, LaMa Apache-2.0, GFPGAN Apache-2.0. Lucy 2.5 is
a Decart model; not affiliated with Decart.
