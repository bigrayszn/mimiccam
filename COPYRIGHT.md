# Copyright

**MimicCam © 2026 Ray Bekson (bigrayszn) — <https://github.com/bigrayszn/mimiccam>.**

MimicCam — the application, its name, its documentation and its original source — is
the work of Ray Bekson. All rights in that work are his.

**Licence: GPL-2.0-or-later.** The full terms are in [LICENSE](LICENSE), which is the
verbatim GNU General Public License v2 (this file is deliberately separate so that the
repository still declares "GPL-2.0" rather than an unknown licence). In short:

- You may run, study, share and modify MimicCam, for any purpose, including
  commercially.
- If you distribute it — or a modified version — you must ship this notice and the
  licence text, make the corresponding source available, and state what you changed.
- The copyright and the original authorship stay with Ray Bekson. A fork does not
  become your project; it is a fork of his, and it has to say so.
- The name **MimicCam** and the repository URL above identify the original. Do not
  present a modified build as the original, and do not use Ray Bekson's name to
  endorse it.

## Where this notice appears

| place | what it says |
|---|---|
| [README.md](README.md) | first paragraph, above the download link |
| [INSTALL.md](INSTALL.md) | first two lines |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | the project copyright, and that components keep their own |
| `mimiccam/__init__.py` | `__author__`, `__copyright__`, `__url__`, `__license__` — the single source the app reads |
| `MimicCam.exe --version` | prints the version, the copyright and the repository |
| the app window | `Copyright (C) 2026 Ray Bekson (bigrayszn)` in the header, repository in its tooltip |
| `installer/*.iss` | publisher is Ray Bekson (bigrayszn); `AppUrl` is the repository |
| every GitHub release | the note opens with this notice |
| the GitHub repository | the description names the author |

`tests/test_core.py` asserts the strings above, so a later edit cannot quietly remove
the attribution.

## Third-party components

Bundled or downloaded models and libraries keep their own copyrights and licences —
YuNet (MIT), InsightFace (non-commercial research), RVM (GPL-3.0), LaMa (Apache-2.0),
GFPGAN (Apache-2.0), PySide6 (LGPL-3.0), OpenCV (Apache-2.0), NumPy (BSD-3-Clause),
Pillow (MIT-CMU), keyring (MIT), pyvirtualcam (GPL-2.0), Decart SDK (MIT), LiveKit
(Apache-2.0). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Nothing in this
file claims any of them.

Lucy 2.5 is a hosted model operated by Decart. MimicCam is not affiliated with Decart,
with OBS, or with the LucidCam project.
