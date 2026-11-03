# MimicCam 0.1.1 — the full-body character swap

This release fixes the thing that made the app feel like a face swap: **Lucy 2.5 was instructed to keep your body and clothes.** It now replaces the whole person — face, hair, beard, build and outfit — and keeps your movement.

## What actually changed

The character preset used to read *"…keep their face, hair, skin tone, body shape and clothing consistent in every frame"*. Decart's prompt guide warns about exactly that wording: asking for a face while asking to keep the clothes produces a new face on the source body. Two 14-second live sessions against the same performer (a woman in a rust T-shirt, with tattoos) and the same character (a man in a black suit):

| measured | 0.1.0 prompt | 0.1.1 prompt |
|---|---|---|
| face similarity to the character | 0.533 | **0.661** |
| dark tailored fraction of the torso | 0.05 | **0.96** |
| colourful fraction (the source's T-shirt) | 0.60 | **0.00** |
| result | face paste, clothes unchanged | **whole-person swap** |

The rust top, the tattoos and the source hairstyle are gone; the suit, shirt and tie are the character's.

- **Whole-person prompt template** plus the character's own description, clamped to Decart's ~750-character limit at a sentence boundary. Saved prompts from 0.1.0 are migrated on launch, because the old wording is the bug.
- **Look notes** — a new field for what the character wears, head to toe: *"black suit jacket, white shirt, black tie, black trousers, black leather shoes."* A chest-up character photo cannot contain trousers, so they are described instead of silently staying yours.
- **Reference conditioning** — the photo is padded to 16:9 instead of cropped, kept between 512 px and 1280 px, transparency flattened, lightly sharpened. The app reads its framing and tells you when a chest-up crop limits the swap.
- **`--check-lucy`** — prints the exact prompt Lucy would receive, free, without opening a billed session.
- **Public installer** — the setup program now ships the app only (no personal character photos or settings); the free local models are downloaded from inside the app on first use.
- 117 unit tests pass, 15 of them pinning this behaviour.

## Install

### Requirements

- **Windows 10 or 11, 64-bit** (tested on Windows 11).
- **NVIDIA GPU and a current NVIDIA driver** for Lucy mode — tested on an RTX 4070 Laptop 8 GB with driver 610.47. Any RTX 20-series or newer is comfortable. The app refuses Lucy mode without an NVIDIA adapter because decode, colour conversion and the virtual camera are pinned to it.
- **OBS Studio** (free, <https://obsproject.com/download>) for its *OBS Virtual Camera* driver. You never have to open OBS; the driver only has to be installed.
- **A webcam** — laptop or USB.
- **A Decart account with credit** for Lucy mode: <https://platform.decart.ai>. Billing is **$0.02 per second** of connected time ($1.20/min); *fast mode* is $0.04/s and only reduces latency.
- Internet: 3.5 Mbps each way minimum.

No Python, no CUDA toolkit, no admin rights, no Visual Studio.

### Option A — installer (recommended)

1. Download `MimicCam-0.1.1-setup.exe` below.
2. Run it. SmartScreen warns because the build is unsigned — **More info → Run anyway**.
3. Keep the default folder and finish. The installer installs per-user to `%LOCALAPPDATA%\Programs\MimicCam` (no UAC), and warns you if no virtual camera is present.
4. If it warned about the camera: install OBS Studio, then start MimicCam again.

### Option B — portable ZIP

1. Download `MimicCam-0.1.1-win64-portable.zip`.
2. Extract to a **short** path, e.g. `C:\MimicCam` — Windows cannot load the virtual-camera DLL from a very deep folder.
3. Run `C:\MimicCam\MimicCam.exe`. Settings and models live in `%APPDATA%\MimicCam`.

### Option C — the Python source (no install)

Download `MimicCam-0.1.1-source.zip`, or clone this repository and follow `INSTALL.md`:

```bat
git clone https://github.com/bigrayszn/mimiccam.git
cd mimiccam
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m mimiccam
```

## Use it with Decart in six steps

1. **Get a key**: <https://platform.decart.ai> → add credit → create an API key (`dct_...`).
2. **Save it**: MimicCam → *Settings* → *Engine* → *Decart API key* → paste → **Save**. It goes to Windows Credential Manager, never a config file. Enter your **balance** (e.g. `10.00`) and press **Set**; Decart has no balance API, so the app subtracts what it meters.
3. **Pick the character photo**: *Character photo…* → a sharp, front-facing, single-subject image. A **full-body** photo is best; a chest-up one is fine if you fill the **Look notes**.
4. **Write the look notes**: everything the character wears, head to toe — `a black suit jacket, white shirt, black tie, black trousers, black leather shoes`.
5. **Start**: engine *Lucy 2.5 via Decart*, then **Start camera**. The preview switches to *output (what callers see)*.
6. **Join the call**: Zoom / Teams / Meet / Discord → camera → **OBS Virtual Camera**. Start MimicCam *before* joining, or reselect the camera.

The credit box shows the rate, billed time, this run's cost and what is left. The app ends the session after 45 s with nothing moving (never in the first 90 s) and reconnects on the next movement — the only way to stop the meter, because Decart's API has no pause call.

**If the clothes still do not change:** you are not on the *Character swap* preset, or the prompt has been edited and lost the phrase *"whole person, head to toe… complete outfit"*. Run `MimicCam.exe --check-lucy --reference character.jpg --notes "…"` to see exactly what is being sent.

## Free local mode (no credits, offline)

The same app ships a local engine: face swap, hair from the character photo, skin-tone harmonisation, optional room rebuild — all on your own GPU with a one-time ~908 MB model download and no internet afterwards. It cannot replace your clothes or body; that is what Lucy mode is for. The InsightFace weights it downloads are licensed for non-commercial research use.

## Privacy

- Lucy mode sends your camera video to Decart — the model runs there. Nothing else leaves the machine.
- Local mode sends nothing anywhere.
- The API key lives in Windows Credential Manager; nothing is logged.

## Licensing

GPL-2.0-or-later. Model licences (YuNet MIT, InsightFace research-only, RVM GPL-3.0, LaMa Apache-2.0) are listed in `THIRD_PARTY_NOTICES.md`. Not affiliated with Decart.

Full details, troubleshooting table and call-app notes: `INSTALL.md` in the repository or in the installed folder.
