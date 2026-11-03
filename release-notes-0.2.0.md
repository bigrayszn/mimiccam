# MimicCam 0.2.0 — the character's clothes, in a room that is actually dark

The free engine used to change your face and stop there. 0.2.0 gives it the
character's **clothes**, makes them **follow your body**, and fixes the reason none
of it was visible in a dim room: the person matte was running at a detail level that
lost your torso entirely.

Everything below is measured on the same machine as 0.1.1 — RTX 4070 Laptop 8 GB,
the backlit webcam that made the low-light work necessary, and the same character
photo — not estimated.

## Wear the character's clothes

New checkbox in the free engine: **Wear the character's clothes (from the photo)**.
It is off by default, because it changes what callers see.

- The character photo is matted once, and the cloth below the chin is cut out of it —
  face, hair and skin removed. What is left is the garment: jacket, shirt, tie, lapel
  pin, whatever the character is wearing. 2,244 × 2,330 px of it on the reference used
  here (36% of the photo).
- Each frame that garment is fitted onto **your torso**, not your face: the reference's
  own silhouette gives a quad (the shoulder line, and a row a face-and-a-half below the
  chin, with the width of each), your live matte gives the same quad, and a
  least-squares affine maps one onto the other. Lean or step sideways and the clothes
  come with you.
- The paste is composited inside your own matte with the live skin punched back
  through, so hands, forearms and neck stay yours and still get the character's skin
  tone from the body pass. Whatever clothing is left outside the pasted outline — your
  shoulders are usually wider than the character's — is **dyed** toward the garment's
  colour with its shading kept.
- It costs **3.2 ms a frame** and rides the matte the room composite already pays for.

| measured | pinned to the face | tracking the torso |
|---|---|---|
| collar position, torso shifted 70 px from under the head | 141 px off | **18 px off** |
| live loop, alternating runs | 31.7 fps, 25 ms | **32.0 fps, 27 ms** |

What it does not do: change your body shape. The quad follows the silhouette, not the
skeleton, so a raised arm carries the sleeve with the outline rather than bending the
elbow. That is still a generative job, and still Lucy's.

## A matte that survives a dark room

The new wardrobe made an old problem impossible to ignore. Robust Video Matting's
`downsample_ratio` decides how much of the frame its encoder reads, and the shipped
0.25 dropped a backlit body almost completely. On the real frame (face at luma 23) the
torso came out at alpha **0.38** at 0.25, **0.47** at 0.5 and **0.86** at 1.0.

So the matte now has two gears, chosen from the light: four dark frames earn the fine
pass, thirty bright ones give it back, and a flickering exposure settles on neither.
A ratio change resets RVM's recurrent state (its state tensors are shaped by the
ratio), which is why the hysteresis exists at all.

In the live loop on that frame the wardrobe went from torso alpha **0.47 to 0.89** —
the suit's shoulders and both lapels appear where a strip did before — at **31.7 vs
32.0 fps** and 25 vs 27 ms, because the matte has its own thread and the spare time was
already there.

The low-light lift from 0.1.1 is unchanged and still does the face: gamma 0.41 takes a
face at luma 23 to 95, and the swap scores **+0.341** against the character where the
unlifted frame scored +0.234.

## Also in this release

- **Torso quad and its fallbacks** — when the silhouette below the chin is unusable
  (dark room, an arm across the chest, nobody there) the wardrobe falls back to scaling
  by face width and anchoring the chins, with a sanity bound against the face ratio so
  an arm is not mistaken for a body.
- **Research notes, with the numbers that killed the ideas** — [FACE_ENHANCERS.md](FACE_ENHANCERS.md)
  records why tiled 256/384/512 px pixel-boost and an fp16 swap network were rejected
  (identity collapsed to -0.04 and -0.072 against +0.65), and the README records why the
  JoyAI 16B video model and the 24B image model cannot run in this loop on 8 GB
  (~21.5 GiB resident / ~28 GiB peak for 480p24, against 8 GB here).
- **201 unit tests** (`python -m pytest tests/ -q`), 18 of them for the wardrobe and 19
  for the low-light path and the matte's gear choice.

## Install

### Requirements

- **Windows 10 or 11, 64-bit.**
- Any **DirectX 12 GPU** for the free engine; an **NVIDIA GPU** for Lucy mode (tested on
  an RTX 4070 Laptop 8 GB, driver 610.47).
- **OBS Studio** (<https://obsproject.com/download>) for its *OBS Virtual Camera* driver.
  You never have to open OBS.
- **A webcam.**
- A **Decart account with credit** for Lucy mode (<https://platform.decart.ai>), at
  $0.02/s. The free engine needs no key and no network after its one-time model download.

### Portable ZIP (what this release attaches)

1. Download `MimicCam-0.2.0-win64-portable.zip` and `SHASUMS256.txt` below.
2. Extract to a **short** path, e.g. `C:\MimicCam` — Windows cannot load the
   virtual-camera DLL from a very deep folder.
3. Run `C:\MimicCam\MimicCam.exe`. Settings and models live in `%APPDATA%\MimicCam`.

The installer is built from this repository (`installer/public.iss` for the app-only
setup, `installer/setup.iss` for the single-file one) — see the header of each for the
two commands.

### From source

```bat
git clone https://github.com/bigrayszn/mimiccam.git
cd mimiccam
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m mimiccam
```

## Getting the wardrobe to look right

1. Load a character photo with the clothes you want, **front-facing and well lit**. A
   chest-up crop still works; the garment is cut where the photo ends.
2. Tick **Wear the character's clothes (from the photo)**.
3. **Light the room.** The paste is gated by your person matte, so a brighter frame means
   a more opaque garment — 0.89 alpha in a lifted frame against 0.47 in a backlit one is
   the difference between a suit and a ghost of one.
4. Sit roughly as the character is photographed: front-on. A turn beyond about 30
   degrees flattens a garment a still photo cannot fill back in.

## Privacy

- The free engine sends nothing anywhere. The models download once.
- Lucy mode sends your camera video to Decart; nothing else leaves the machine.
- The API key lives in Windows Credential Manager.

## Licensing

GPL-2.0-or-later. Model licences: YuNet MIT, InsightFace research-only, RVM GPL-3.0,
LaMa Apache-2.0, GFPGAN Apache-2.0 — see `THIRD_PARTY_NOTICES.md`. Not affiliated with
Decart.
