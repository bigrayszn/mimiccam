# Face enhancers: what can legally ship, and what fits in a frame

inswapper renders a **128 px** face and MimicCam pastes it at webcam scale, so the
swapped face is the softest thing on screen. This is the licence and speed survey
for a post-pass that adds detail, written so the decision does not have to be made
from memory again.

Measured on the machine this project is developed on: **RTX 4070 Laptop, 8 GB,
driver 610.47**, `onnxruntime 1.23.0` with `DmlExecutionProvider`, fp32 IO,
ONNX Runtime graph optimisation enabled, 10 timed runs after 3 warm-up runs.

## The candidates

| Model | Licence on the release | ONNX size | Measured (DirectML) | Verdict |
|---|---|---|---|---|
| **GFPGAN v1.4** (TencentARC) | **Apache-2.0** - `LICENSE` says "GFPGAN is licensed under the Apache License Version 2.0 except for the third-party components listed below"; the list is `basicsr` (Apache-2.0), `numpy` (BSD-3), `tqdm` (MIT). No dataset clause. | 340 MB fp32 | **44.3 ms** at 512x512 fixed input, 23 fps of its own | **Shippable**, at render cadence rather than per frame |
| **Real-ESRGAN `realesr-general-x4v3`** (xinntao) | **BSD-3-Clause** - `xinntao/Real-ESRGAN` repo licence, weights released by the same project | 4.87 MB | **6.1 ms** 128->512, **17.5 ms** 256->1024 | **Shippable and cheap**, but a general upscaler: no face prior |
| CodeFormer (sczhou) | S-Lab License 1.0 - **non-commercial** | - | - | Excluded |
| GPEN (yangxy) | **No licence file at all** (GitHub reports none) | - | - | Excluded: all rights reserved |
| FaceFusion's exported `gfpgan_1.4.onnx` / `gpen_bfr_*` / `codeformer.onnx` | **OpenRAIL-AS** use-restricted | - | - | Excluded: not compatible with this project's licence, whatever the upstream model says |
| SwinIR (JingyunLiang) | Apache-2.0 | - | not measured; window attention at 512 px is far heavier than the two measured above | Shippable, but a general restorer with no face prior |
| InstructIR (mv-lab) | MIT | - | - | Shippable, designed for photo restoration rather than realtime |

## What the two licences mean here

MimicCam is `GPL-2.0-or-later`. **Apache-2.0 is compatible with GPL-3.0 but not
with GPL-2.0-only**; because the project says "or later", a distributor can take
the whole under GPL-3.0 and bundle Apache-2.0 weights, with the attribution and
NOTICE obligations satisfied in `THIRD_PARTY_NOTICES.md`. BSD-3-Clause is
GPL-compatible without that condition, which makes Real-ESRGAN the simpler of the
two to argue about.

The one honest caveat on GFPGAN: it was trained on **FFHQ**, whose dataset terms
are CC BY-NC-SA 4.0. Tencent's licence grants Apache-2.0 over the release and says
nothing about the dataset, so the operative grant is permissive; the provenance is
still a judgement call. Note that the weights this project already ships -
`inswapper_128.onnx`, `w600k_r50.onnx` - are **explicitly non-commercial
research**, so GFPGAN is not the strictest licence in the bundle; it is the
loosest model in it.

## What the measurements say about fitting them in a frame

- **GFPGAN is 44 ms** at its fixed 512x512 input. At 30 fps a frame is 33 ms, so it
  cannot run on every frame beside the swap network. It fits the *render* cadence
  instead: the engine refreshes the swap texture on its own thread and cross-fades
  it (`TEXTURE_BLEND`), measured at **23 refreshes/s** without the enhancer and
  **18/s** with it (the pass itself runs on every second render, about 9 Hz), so
  it rides the path that already tolerates a slower texture and costs under half a
  core-second per second.
- **Real-ESRGAN x4v3 is 6.1 ms** for a 128 px crop, i.e. it fits inside a 33 ms
  frame with room to spare, even every frame.
- Both take fp32 NCHW. GFPGAN's ONNX expects BGR in `[-1, 1]`: feeding RGB produced
  visibly wrong skin tone on the first try, BGR matched the source.
- Quality, on a real 128 px crop upscaled to 512: GFPGAN visibly rebuilt eyes,
  beard strands and skin texture that bicubic interpolation left flat. Real-ESRGAN
  sharpens edges but invents no facial structure - it will amplify inswapper's own
  artefacts as readily as it sharpens them.

## Recommendation

1. **GFPGAN v1.4** as an optional *Detail* pass, applied to the rendered 128 px
   texture every N-th render and blended like the current texture updates. Apache-2.0
   with attribution, 340 MB in the model registry, ~44 ms each time it runs.

   Shipped as described, with one correction the measurement forced: the enhanced
   texture has to be **pasted at 512 px**. Downscaling the enhancer's output back to
   the swap network's 128 px before compositing recovers almost none of the gain -
   measured 7% more sharpness in the frame - while pasting it as rendered gives
   **3.4x** (Laplacian variance 10.3 -> 35.1) for 1.7 ms more per frame. Everything
   that keys off the texture size (the paste transform, the expression masks) scales
   from it, so this needs no change to the swap network.
2. **Real-ESRGAN x4v3** if a cheaper, unconditionally safe option is wanted: 4.87 MB,
   BSD-3, 6 ms, sharpening only.
3. Do **not** ship CodeFormer, GPEN, or any FaceFusion-exported ONNX: the first is
   non-commercial, the second has no licence, and the third carries OpenRAIL-AS use
   restrictions.
