"""Lucy virtual try-on: replace what the person is wearing, from a garment photo.

``lucy-vton-latest`` is a different realtime model from the character swap. It
takes the live camera plus a **garment** reference image - a shirt or jacket on
a clean background, not a face - and streams back the person wearing it. Decart's
guidance is specific, and everything here follows it:

* the reference is the clothing item on its own, at least 512 px, no model in it;
* the prompt is one garment instruction of 20-30 words naming colour, material
  and fit, phrased as *substitute* (replacing something) or *add* (something new);
* one garment at a time - two tops in one sentence is how the output turns to
  mush.

Image and text work only, so this module is testable without an API key.
"""

from __future__ import annotations

from pathlib import Path

# The realtime try-on model. Namespaced through decart.models.realtime(), which
# also knows lucy-vton-3.5 / 3.6 if this ever needs pinning to a revision.
MODEL = "lucy-vton-latest"

MIN_SHORT_SIDE = 512
MAX_LONG_SIDE = 1280
# The sentence openers the model was trained to read as an edit instruction.
VERBS = ("substitute", "add", "change", "replace", "put", "swap", "dress", "wear")
DEFAULT_PROMPT = (
    "Substitute the current top with the garment in the reference image, "
    "matching its fit, fabric and colour"
)


def as_prompt(text: str) -> str:
    """One garment instruction, in the shape the model was trained on.

    A bare description ("black leather bomber jacket") is what a user types, and
    the model reads it as a caption rather than an edit. Prefixing the documented
    verb turns it into an instruction; a sentence that already carries one is
    left exactly as written, because the user's own wording is more specific than
    anything this module could add.
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return DEFAULT_PROMPT
    if cleaned.lower().startswith(VERBS):
        return cleaned
    return f"Substitute the current top with {cleaned}"


def prepare_garment(path: str | Path) -> bytes:
    """Garment photo -> JPEG bytes the try-on model can use as its reference.

    Unlike a character photo this is not padded to the output aspect: the model
    reads the garment itself, so blank bars around it would only shrink it. The
    short side is raised to 512 px because the model reproduces what it sees, and
    a thumbnail of a jacket produces a blurry jacket. Transparent cut-outs are
    flattened onto white rather than black, which is what the model expects of a
    product shot.
    """
    from PIL import Image

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(str(path))

    image = Image.open(source)
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        image = flat
    else:
        image = image.convert("RGB")

    width, height = image.size
    short_side = min(width, height)
    if short_side < MIN_SHORT_SIDE:
        scale = MIN_SHORT_SIDE / max(1, short_side)
        image = image.resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.LANCZOS,
        )
    if max(image.size) > MAX_LONG_SIDE:
        image.thumbnail((MAX_LONG_SIDE, MAX_LONG_SIDE), Image.LANCZOS)

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=93)
    return buffer.getvalue()
