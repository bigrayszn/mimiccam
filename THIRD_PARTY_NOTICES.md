# Third-party notices

**MimicCam © 2026 Ray Bekson (bigrayszn)** — <https://github.com/bigrayszn/mimiccam>.
That copyright covers MimicCam itself. The components below keep their own
licences and copyrights, and nothing here claims them.

MimicCam is distributed under GPL-2.0-or-later; the "or later" clause is what
makes the Apache-2.0 GFPGAN row below compatible, and GPL-3.0 is the version
that is actually taken. It depends on the following projects; include their
license texts in any binary release.

| Component | License | Notes |
|---|---|---|
| [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) | GPL-2.0 | Frames are written to the virtual camera through its shared-memory backends. This is why MimicCam binaries are GPL-2.0 as a whole. |
| [Decart Python SDK](https://github.com/DecartAI/sdk) | MIT | Realtime WebRTC client for Lucy 2.5. |
| [LiveKit Python SDK](https://github.com/livekit/python-sdk) | Apache-2.0 | WebRTC transport used by the Decart realtime SDK. |
| [PySide6 / Qt for Python](https://www.qt.io/qt-for-python) | LGPL-3.0 (Qt) | GUI toolkit. Dynamic linking only; do not statically link. |
| [OpenCV](https://opencv.org) | Apache-2.0 | Capture, colour conversion, resizing. |
| [NumPy](https://numpy.org) | BSD-3-Clause | Frame buffers. |
| [Pillow](https://python-pillow.org) | MIT-CMU | Reference-image normalisation. |
| [keyring](https://github.com/jaraco/keyring) | MIT | Windows Credential Manager access. |
| [Unity Capture](https://github.com/schellingb/UnityCapture) | MIT (filter) / zlib (Unity plugin) | Optional virtual-camera backend for machines without OBS. Not redistributed by this repository; see `installer.iss`. |
| OBS Studio | GPL-2.0-or-later | Optional external dependency providing the OBS Virtual Camera device. Not redistributed. |
| [GFPGAN v1.4](https://github.com/TencentARC/GFPGAN) | Apache-2.0 | Optional Detail pass over the swapped face. The release is licensed Apache-2.0 by Tencent; the weights were trained on FFHQ, whose dataset terms are CC BY-NC-SA 4.0. Downloadable, not bundled; see `FACE_ENHANCERS.md`. |

Lucy 2.5 is a hosted model operated by Decart. Using it is subject to Decart's
terms and pricing; the model weights are not distributed with this project.
