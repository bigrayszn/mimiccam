# MimicCam 0.2.3 — install and use

**MimicCam © 2026 Ray Bekson (bigrayszn)** — <https://github.com/bigrayszn/mimiccam>.
Open source under GPL-2.0-or-later; the copyright stays with Ray Bekson.

MimicCam turns your webcam into a **character camera**: you appear in Zoom, Teams,
Google Meet, Discord, OBS or Streamlabs as somebody else, live, while keeping your
own head movement and expressions. Other apps see it as a normal camera called
*OBS Virtual Camera*.

Two engines ship in the same app:

| | **Lucy 2.5 (Decart)** | **Local model (free)** |
|---|---|---|
| What it replaces | **The whole person** — face, hair, beard, body shape, clothes | Face only, plus hair where the character photo has it, plus skin tone on neck/arms |
| Where it runs | Decart's data centre, over WebRTC | Your NVIDIA GPU (DirectML) |
| Cost | **$0.02 per second** of connected time ($1.20/min), $0.04/s in fast mode | Free, forever |
| Internet | Required, ~2 Mbps up and down | Not required after the one-time model download |
| Models to download | None | ~908 MB, once |
| Best for | Full character swap, costume, hairstyle, body | Face swaps on a budget, offline use |

This guide covers both. If you want what people mean by "Lucy 2.5 quality" — a
full-body clone with the clothes and hair — that is the **Lucy 2.5** mode and it is
the default in this build.

---

## 1. What you need

### Required for every mode

- **Windows 10 or Windows 11, 64-bit.** Windows 10 1809 or newer; Windows 11 is
  what this build was tested on.
- **A webcam.** Any DirectShow camera works: laptop camera, USB webcam, or a phone
  used as a webcam (DroidCam, Iriun). 720p is what goes on the wire; 1080p cameras
  are downscaled automatically.
- **OBS Studio** (free), for its virtual camera driver:
  <https://obsproject.com/download>.
  The app writes its output into *OBS Virtual Camera*; Zoom and friends then see
  that camera. You do **not** have to run OBS or configure anything in it — the
  driver only has to be installed. OBS Studio 28 or newer includes it by default.
- **~400 MB of disk** for the application, plus ~908 MB more if you use the local
  (free) engine.

### Required for Lucy 2.5 mode

- **An NVIDIA GPU.** This is enforced by the app: Lucy mode refuses to start
  without one, because the decode, colour conversion and virtual-camera feed are
  pinned to the NVIDIA adapter and the alternative is a stuttering call.
  - Tested on: **RTX 4070 Laptop, 8 GB, driver 610.47**.
  - Any RTX 20-series or newer is comfortably enough (GTX 16-series works too but
    is not what was measured).
  - **An up-to-date NVIDIA driver** — install *NVIDIA Studio Driver* or *Game Ready
    Driver* from <https://www.nvidia.com/download/>. The bundled ONNX runtime uses
    **DirectML**, which comes with the driver; there is nothing else to install.
    A driver older than roughly 2023 will fail to expose a DirectX adapter and the
    engine will say so.
  - Integrated graphics (Intel/AMD) and Apple/AMD-only laptops cannot run Lucy
    mode in this build.
- **A Decart account with credit.** See step 4.
- **A stable internet connection**: 3.5 Mbps per direction minimum, 5 Mbps
  recommended. Decart's own guidance is round-trip latency ≤150 ms good, ≤300 ms
  fair.

### Not required

- No Python, no Visual Studio, no CUDA toolkit, no admin rights (the installer is
  per-user), no OBS configuration.

---

## 2. Install (recommended: the setup program)

1. Go to the **Releases** page of this repository and download
   **`MimicCam-0.2.3-setup.exe`** (or the portable zip from the same release).
2. Double-click it. Windows SmartScreen will say *"Windows protected your PC"*
   because the build is not code-signed — choose **More info → Run anyway**.
3. Accept the licence (GPL-2.0-or-later), keep the default folder
   (`%LOCALAPPDATA%\Programs\MimicCam`), optionally tick *Create a desktop
   shortcut*, and finish.
4. The installer checks whether a virtual camera exists. If it does not, it tells
   you to install OBS Studio and shows the link. Install OBS Studio, then start
   MimicCam again.

Do **not** move the installed folder to a very deep path (`C:\Users\you\Documents\
stuff\apps\video\tools\MimicCam` and deeper). Windows has a ~260-character limit on
DLL paths and the virtual-camera driver sits deep inside the bundle; the installer
already picks a short per-user location for this reason.

### Alternative: portable ZIP

Download **`MimicCam-0.2.3-win64-portable.zip`**, extract it to a **short** path such
as `C:\MimicCam`, and run `C:\MimicCam\MimicCam.exe`. Nothing is written to the
registry; settings and downloaded models live in `%APPDATA%\MimicCam`.

### Alternative: run from source (no installer at all)

Everything the .exe contains is in this repository. Python **3.10, 64-bit** is what
it is developed and tested against.

```bat
git clone https://github.com/bigrayszn/mimiccam.git
cd mimiccam

py -3.10 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

:: start the app
python -m mimiccam

:: or the one-line launcher that does the venv + run for you
run.bat
```

Requirements are `PySide6`, `opencv-python`, `numpy`, `onnxruntime-directml`,
`pyvirtualcam`, `keyring`, `livekit`, `decart`, `pillow`, `av` (see
`requirements.txt`; `onnxruntime-directml` is what puts the GPU to work).

Useful command-line switches, same in source and in the frozen exe:

```bat
python -m mimiccam --list-cameras        :: probe cameras and their real resolution
python -m mimiccam --list-backends       :: is a virtual camera installed?
python -m mimiccam --check-local         :: load the offline models, swap one frame, exit
python -m mimiccam --check-lucy --reference me.jpg --notes "black suit, black shoes"
                                         :: print the exact prompt Lucy would receive (free)
python -m mimiccam --selftest 15 --camera 1 --mode local --reference me.jpg
```

The verification scripts used to produce the numbers in `README.md` live in
`tests/` and run the same way (`python tests/verify_motion.py`,
`python tests/verify_decart.py --live --quality performer.jpg`).

To build the exe and the installer yourself:

```bat
pip install pyinstaller
python build.py --dist C:\MimicCam-dist      :: portable folder + zip + smoke tests
iscc installer\public.iss                    :: setup .exe (needs Inno Setup 6)
```

---

## 3. Install the virtual camera (OBS Studio)

1. Install OBS Studio: <https://obsproject.com/download>. Default options are fine.
2. You do not need to open OBS. But if MimicCam says *"no virtual-camera backend"*,
   open OBS once, press **Start Virtual Camera**, then **Stop Virtual Camera**.
   That registers the driver with Windows; OBS can be closed afterwards.
3. Check from a terminal:

   ```bat
   MimicCam.exe --list-backends
   ```

   Expected: `Available virtual-camera backends: obs`. If it prints `none`, the
   driver is not registered — reinstall OBS Studio, or use *Unity Capture*
   (advanced, see `installer/vcam_check.issinc`).

Zoom, Teams, Meet and Discord all enumerate cameras when they start. **Start
MimicCam first, then join/open the call**, or reselect the camera in the call's
settings.

---

## 4. Get a Decart API key (Lucy 2.5 mode)

1. Create an account at <https://platform.decart.ai> (Decart's developer platform).
2. Open the dashboard and **add credit**. $10 is about 8 minutes of standard
   Lucy 2.5 time; there is no free tier and no monthly plan — it is pure
   pay-per-second.
3. Create an **API key**. It looks like `dct_...` and is shown once; copy it.
4. In MimicCam: **Settings → Engine → Decart API key**, paste, press **Save**.
   The key is stored in **Windows Credential Manager**, never in a config file, and
   the field shows only the first and last four characters afterwards.
   *(Power users: the `DECART_API_KEY` environment variable overrides the stored
   key and is handy for CI.)*
5. In the same panel, type your **balance** in dollars (e.g. `10.00`) and press
   **Set**. Decart publishes no balance API, so MimicCam cannot read it; it
   subtracts the seconds it meters from the figure you enter. Re-set it after a
   top-up or after spending with another tool.

Costs, exactly:

| | standard | fast mode |
|---|---|---|
| rate | $0.02 / s | $0.04 / s |
| per minute | $1.20 | $2.40 |
| 10 minutes | $12.00 | $24.00 |

Billing is **connected time**, not frames, and it does not depend on resolution.
Fast mode buys lower latency, not a better picture — which is why the checkbox says
so. The app ends the session automatically after 45 s with nothing moving in front
of the camera (never inside the first 90 s), shows your raw camera while paused,
and reconnects on the next movement: that is the only way to stop the meter,
because the API has no pause call.

---

## 5. Choose your character photo (the part that decides quality)

The character photo is the character. Everything the model knows about the new
person comes from it plus the words in the prompt.

**Best:** a **full-body, head-to-toe photo** of the character, front-facing,
evenly lit, one person, nothing covering the face, hands empty, plain-ish
background. Trousers, shoes and silhouette then come from the photo.

**Good:** a waist-up or chest-up photo — very common in practice. The face, hair,
jacket and tie will be perfect; the legs will not exist in the photo, so they have
to be described:

- Click **Character photo...** and pick the image.
- Fill in **Look notes**, e.g.
  `a black suit jacket, white shirt, black tie, black trousers, black leather shoes`.
- MimicCam tells you which case you are in: *"Chest-up reference: Lucy knows the
  character's face and jacket, not their trousers or shoes."* That message is
  generated by reading the photo, not by guessing.

Practical rules that measurably matter:

- **≥512 px on the short side**, ideally ~1280 px on the long side. Smaller
  photos are upscaled; blurry ones produce a blurry character.
- **Match the framing**: a full-body reference for a full-body shot.
- **One person in the photo.** A second face makes the model pick one.
- **No heavy props** in frame (glasses in hand, a guitar, a trophy) — they get
  projected into the output.
- **Avoid drastic build mismatches** (a 150 cm reference on a 195 cm performer);
  the model warps rather than redraws the skeleton.
- PNG, JPEG or WebP. Transparency is flattened onto the photo's own blurred
  backdrop, and the image is padded (never cropped) to 16:9 before it is sent.

MimicCam prepares the file for you: pad to the output aspect, 512–1280 px,
light sharpening, JPEG. You can see exactly what will be sent — including the
prompt — with:

```bat
MimicCam.exe --check-lucy --reference "C:\path\character.jpg" --notes "black suit, black shoes"
```

---

## 6. Run it

1. Start **MimicCam**.
2. **Camera** group → pick your webcam. If the wrong one is chosen, `Refresh` and
   look for the index that shows a picture (index 0 is sometimes an IR camera).
3. **Engine** → **Lucy 2.5 via Decart** (this build starts there once a key is
   saved). Leave *Fast mode* off unless you need the lowest latency.
4. **Character** → choose the photo, fill the **Look notes**.
5. Press **Start camera**. The preview switches to *output (what callers see)*.
6. Open Zoom/Teams/Meet/Discord, choose **OBS Virtual Camera**, join.

The status chips under the preview show the virtual-camera backend, the frame rate
and the Lucy round-trip latency. The credit box above the preview shows the rate,
the billed time, what this run has cost, and what is left of the balance.

### The look presets

| Preset | What it does |
|---|---|
| **Character swap (use reference image)** | The whole-person substitution. This is the one you want for a clone. |
| Same person, studio lighting | Relights you; keeps you. |
| Anime character | Cel-shaded restyle; keeps your framing. |
| Cinematic film look | Colour grade only. |
| Formal outfit | Dresses you in a suit, keeps your identity. |
| Custom prompt | Whatever you type. |

The character preset is a template, and the wording is not decorative: Decart's own
guide warns that asking for "a face replacement" produces exactly that — a new face
on your body, in your clothes. If you edit the prompt by hand, keep the phrases
*"whole person"*, *"head to toe"* and *"complete outfit"*; drop them and the swap
regresses to a face paste. Prompts over ~750 characters are rejected by the server
(they name how much of yours fit), which is why the app trims at a sentence
boundary.

---

## 7. Free local mode

Tick **Local model (free, offline — no credits)** instead of Lucy. On first use the
app downloads five ONNX files (~908 MB total) into `%APPDATA%\MimicCam\models` and
then works with no internet and no key.

What it can do: swap the face on your live camera with the character's identity,
paste the character's hair where the photo has hair, recolour neck/arms skin, and
optionally rebuild the room from the character photo (*"Seat me in the reference
photo's room"*).

What it cannot do, and never will: replace your clothes or your body. It is a
128-pixel face network plus colour work. If your complaint is "my jersey is still
there", that is this engine's limit, not a bug — use Lucy mode.

The InsightFace weights it downloads are licensed **for non-commercial research
use**. Personal projects: fine. A paid product: get a commercial licence from
InsightFace first. The full list is in `THIRD_PARTY_NOTICES.md` and in the app's
model dialog.

---

## 8. Using it in calls

- **Zoom**: Settings → Video → Camera → *OBS Virtual Camera*. Turn off
  *Mirror my video* if the character looks flipped (MimicCam mirrors the preview,
  not the stream).
- **Microsoft Teams**: Settings → Devices → Camera → *OBS Virtual Camera*.
- **Google Meet / Discord / Slack huddles / Streamlabs / OBS**: pick
  *OBS Virtual Camera* in the camera dropdown.
- **Recordings and streams**: add *Video Capture Device → OBS Virtual Camera* in
  OBS, or *Video Source → Camera* in Streamlabs.
- The virtual camera stays available while MimicCam runs. Closing MimicCam releases
  it and the call falls back to the "no camera" card.

Latency expectation: Lucy adds a few hundred milliseconds of network + model time
on top of your camera. On the tested RTX 4070 Laptop with a nearby region the
round-trip is typically well under half a second; the app shows the measured value.

---

## 9. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `NVIDIA GPU required` dialog | No NVIDIA adapter, or a driver too old to expose a DirectX device. Update the driver. Integrated graphics cannot run Lucy mode. |
| `Invalid API key` in the log | Key mistyped or revoked. Re-paste it, Save, restart the engine. |
| `insufficient_credits` | Decart balance is empty. Top up; the local meter is not authoritative. |
| Zoom shows "no camera" / a black frame | Start MimicCam *before* the call, then reselect the camera. Verify with `MimicCam.exe --list-backends`. |
| `The system cannot find the file specified` when starting | The exe was moved too deep for Windows to load its DLLs. Keep it in `C:\MimicCam` or the installer's default folder. |
| Picture stutters or repeats frames | Lucy needs 3.5 Mbps each way; check the connection. Fast mode reduces latency at 2x the price. On the local engine, close GPU-heavy apps. |
| Face swaps but the clothes do not | This is the prompt, not the model. Use the **Character** preset and **Look notes**. See section 5 and 6. |
| Character looks nothing like the photo | Photo is too small, blurry, side-on, occluded, or has a second face. Use a sharp, front-facing, single-subject image. |
| Everything is mirrored | Uncheck mirroring in the call app (MimicCam's `Mirror` setting affects only the preview). |
| Local mode refuses to start | It needs the downloaded models and an NVIDIA adapter (DirectML). Run `MimicCam.exe --check-local`. |

Logs are in the window's log pane; nothing is written to a log file, and the API
key is never logged.

---

## 10. Privacy and licensing

- **Lucy mode sends your camera video to Decart.** That is unavoidable — the model
  runs there. Nothing else leaves the machine; the key is in Windows Credential
  Manager, not in any file this repo or the app writes.
- **Local mode sends nothing anywhere.** After the model download it works with the
  network cable unplugged.
- **The virtual camera only receives frames while you press Start.** Nothing is
  recorded by MimicCam itself.
- MimicCam is **GPL-2.0-or-later**. The bundled/open-weight models have their own
  licences: YuNet (MIT), InsightFace `inswapper_128` and `w600k_r50` (non-commercial
  research), Robust Video Matting (GPL-3.0), LaMa (Apache-2.0). Details in
  `THIRD_PARTY_NOTICES.md`.
- MimicCam is not affiliated with Decart. Lucy is a Decart model; you are billed by
  Decart, under their terms.
- Use it on people who know. Impersonating someone to deceive them is illegal in
  most places and is not what this project is for.

---

## 11. Uninstall

- **Installed with the setup program**: Windows Settings → Apps → MimicCam →
  Uninstall (or the Start-menu shortcut). The application is removed.
- **Settings, character photos and downloaded models** live in
  `%APPDATA%\MimicCam` and are left alone so a reinstall keeps them. Delete that
  folder to reclaim the ~908 MB of models.
- **API key**: Windows Credential Manager → *Windows Credentials* → `MimicCam` →
  Remove. Uninstalling does not touch it.
- **Portable zip**: delete the folder you extracted. Same `%APPDATA%` note.
