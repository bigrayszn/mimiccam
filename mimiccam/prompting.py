# SPDX-FileCopyrightText: 2026 Ray Bekson (bigrayszn) <https://github.com/bigrayszn/mimiccam>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Prompt construction for Lucy 2.5.

Decart's own prompting guide is unusually explicit, and every rule below comes
straight from it:

* use the whole-person verb - "Substitute the person in the video with the
  person from the reference image" - never "swap the face", which the model
  obeys by keeping the body and the clothes,
* describe the character in words as well as pixels: "the more detail you
  provide, the closer the output matches the reference",
* say what stays (room, framing, lighting) rather than what must not happen -
  negative instructions are documented as unreliable,
* one focused edit, no filler adjectives, about 750 characters maximum.

The last two rules are why prompts are assembled here instead of being typed by
hand: an over-long prompt is rejected by the server, and an edited one commonly
loses the sentence that makes the clothes change.
"""

from __future__ import annotations

from typing import Optional

from .reference import Appearance, Framing

MAX_PROMPT_CHARS = 750

# The sentence that does the work. "Whole person, head to toe" and the explicit
# clothing clause exist because a character swap that only names a face is
# carried out as a face swap: the model keeps the source wardrobe.
CHARACTER_TEMPLATE = (
    "Substitute the whole person in the video, head to toe, with the person in "
    "the reference image. Their face, hair, body shape and complete outfit "
    "replace those of the person in the video, including the clothes they are "
    "wearing now, so the character's own garments appear from head to toe. The "
    "new face, hair and clothes follow every movement, pose and gesture. Keep "
    "the room, the camera framing and the lighting unchanged."
)

LEG_WORDS = (
    "trouser", "pant", "jean", "short", "skirt", "dress", "shoe", "boot",
    "sneaker", "leg", "foot", "sock", "trainer", "heel", "knee", "cargo",
)


def _sentences(text: str) -> list[str]:
    flat = " ".join(text.replace("\n", " ").split())
    return [part.strip().rstrip(".") for part in flat.split(". ") if part.strip()]


def clamp(text: str, limit: int = MAX_PROMPT_CHARS) -> str:
    """Cut at a sentence boundary the server will accept.

    Decart rejects prompts over roughly 750 characters with an error naming how
    much of it fit. Trimming here keeps the important first sentences and never
    sends a prompt that fails.
    """
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    kept = ""
    for sentence in text.split(". "):
        candidate = f"{kept} {sentence}.".strip() if kept else f"{sentence}."
        if len(candidate) > limit:
            break
        kept = candidate
    if kept:
        return kept
    return text[: limit - 1].rsplit(" ", 1)[0] + "."


def _below_frame(framing: Optional[Framing], notes: str) -> str:
    """The lower half, when the reference photo never showed it."""
    if framing is None or framing.kind not in ("portrait", "half"):
        return ""
    lowered = notes.lower()
    if any(word in lowered for word in LEG_WORDS):
        return ""
    return (
        "Continue the same outfit below the frame: matching trousers and the "
        "matching shoes."
    )


def character_prompt(
    base: str,
    appearance: Optional[Appearance] = None,
    notes: str = "",
    framing: Optional[Framing] = None,
    limit: int = MAX_PROMPT_CHARS,
) -> str:
    """The full-body character swap prompt: template + description + look notes."""
    text = (base or CHARACTER_TEMPLATE).strip()
    if "keep their face, hair, skin tone, body shape and clothing" in text.lower():
        # A prompt saved by an older build: the wording is the bug, so the whole
        # template is replaced rather than extended.
        text = CHARACTER_TEMPLATE
    if not text.lower().startswith("substitute"):
        text = f"{CHARACTER_TEMPLATE} {text}"

    extra: list[str] = []
    if appearance is not None and appearance.hair:
        extra.append(f"They have {appearance.hair} hair.")
    clean_notes = notes.strip().rstrip(".")
    if clean_notes:
        extra.append(f"They wear {clean_notes}.")
    elif appearance is not None and appearance.garments:
        joined = " and ".join(appearance.garments)
        extra.append(f"They wear mostly {joined} clothing.")
    below = _below_frame(
        framing or (appearance.framing if appearance else None), f"{notes} {text}"
    )
    if below:
        extra.append(below)

    if extra:
        sentences = _sentences(text)
        first, rest = sentences[0], sentences[1:]
        blocks = [f"{first}.", " ".join(extra)] + [f"{sentence}." for sentence in rest]
        text = " ".join(blocks)
    return clamp(" ".join(text.split()), limit=limit)


def face_only_note(preset_key: str) -> str:
    """Presets whose wording deliberately preserves the source person."""
    return preset_key in ("identity", "cinematic", "suit")
