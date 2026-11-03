# MimicCam 0.2.3 — the name is reserved, and the binaries prove it

**MimicCam is © 2026 Ray Bekson (bigrayszn) — <https://github.com/bigrayszn/mimiccam>.**
The code stays open source under **GPL-2.0-or-later**. The name, the icon and the
author's identity do not come with it.

0.2.2 made it clear who owns the project. This release makes it hard to take: the
license covers the code, a separate policy covers the brand, every source file carries
the copyright inside it, and every binary now records — and can be verified as — who
published it.

## The name is not part of the license

**[TRADEMARK.md](TRADEMARK.md)** is new. A software license never grants trademark
rights, and GPL-2.0 is no exception, so the mark is now claimed explicitly:

- **Allowed:** say MimicCam, review it, teach with it, link to it, say "based on
  MimicCam", redistribute an unmodified official build under its own name, fork the
  code and publish the fork.
- **Not allowed:** ship a *modified* build as MimicCam or anything confusingly close
  (*MimicCam Pro*, *MimicCam+*, *Mimic-Cam*), reuse the icon for a fork, imply the
  original author endorses it, or register the name for a competing product.
- **A fork must rename**, replace the icon, and keep `LICENSE`, `NOTICE`,
  `COPYRIGHT.md` and the copyright headers — which GPL-2.0 §5 required anyway.

This is the arrangement Firefox, Signal and VLC use: the source is free, the brand is
not, so a fork can never pass itself off as the original.

## The notice travels with the code

- **[NOTICE](NOTICE)** — the short form every redistribution must carry: the
  copyright, what GPL-2.0 obliges a distributor to keep, and that the name is not
  licensed.
- **Every source file** — 54 files now open with
  `SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn)` and
  `SPDX-License-Identifier: GPL-2.0-or-later`. A file copied out of the repository
  still says whose it is, and a machine can read it.
- **The installer ships the paperwork** — `LICENSE`, `NOTICE`, `COPYRIGHT.md` and
  `TRADEMARK.md` land next to the exe, so a copy found on a machine years later still
  names its owner.
- **[SIGNING.md](SIGNING.md)** — what a certificate costs, how to wire it in, and the
  two registrations (copyright, trademark) that turn a notice into a remedy.

## The binaries say who built them

- **Explorer → Properties → Details** on `MimicCam.exe` now reads
  `Company Name: Ray Bekson (bigrayszn)`, `Legal Copyright: Copyright (C) 2026 Ray
  Bekson (bigrayszn) - GPL-2.0-or-later`, product name, version and the repository URL.
  The version resource is generated from `mimiccam/__init__.py`, so it cannot drift.
- **Code signing is wired in.** Set `MIMICCAM_SIGN_PFX` + `MIMICCAM_SIGN_PASSWORD`, or
  `MIMICCAM_SIGN_THUMBPRINT`, or any `MIMICCAM_SIGN_COMMAND` for a cloud signer, and
  `python build.py` signs the exe and the setup, timestamps both, and re-verifies them
  with `signtool verify /pa` — failing the build if a signature does not hold. With no
  identity set, the build is exactly what it was before and says so.
- Tested end to end on a throwaway self-signed certificate: exe signed, setup signed,
  subject `CN=Ray Bekson (bigrayszn)`, both timestamps valid, verification error
  correctly downgraded to a warning only for an untrusted root.
- The setup is signed after Inno compiles it, because Inno's `SignTool` directive
  cannot be driven from a command line that has spaces in it. The uninstaller stays
  unsigned; the exe and setup — the files a user downloads — do not.

## What this does and does not stop

It cannot stop anyone from copying the code: that is what open source means, and
GPL-2.0 allows the copy as long as the notice and the source travel with it. What it
stops is a copy passing as **yours** — a fork can only appear under its own name and
its own publisher, because nobody else can produce a signature that says Ray Bekson.

The next steps that need a filing rather than a commit are listed in
[SIGNING.md](SIGNING.md): copyright registration (statutory damages) and a trademark
application for the word mark (®).

## Verified

- **211 unit tests** pass (`python -m unittest discover -s tests -v`), six of them new:
  the name policy, the NOTICE, the documents the installer ships, the header on all 54
  files, the exe's version resource, and that signing stays opt-in and reads the
  identity from the environment.
- The frozen exe, the installer and the window were all exercised: `--version` prints
  the copyright and the repository, the UI verification script passes end to end.

## Install

Download `MimicCam-0.2.3-win64-portable.zip` and `SHASUMS256.txt` below, extract to a
**short** path (e.g. `C:\MimicCam` — Windows cannot load the virtual-camera DLL from a
very deep folder) and run `MimicCam.exe`. Windows 10 or 11, 64-bit; any DirectX 12 GPU
for the free engine, an NVIDIA GPU for Lucy mode; OBS Studio for its virtual-camera
driver. Full instructions: [INSTALL.md](INSTALL.md).

## Licensing

GPL-2.0-or-later, © 2026 Ray Bekson (bigrayszn). The name and branding are governed by
`TRADEMARK.md` and are not granted by the license. Model licenses: YuNet MIT,
InsightFace research-only, RVM GPL-3.0, LaMa Apache-2.0, GFPGAN Apache-2.0 — see
`THIRD_PARTY_NOTICES.md`. Lucy 2.5 is a Decart model; not affiliated with Decart.
