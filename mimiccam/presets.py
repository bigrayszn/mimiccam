"""Prompt presets for Lucy 2.5.

Decart's guidance is to describe what should change in observation terms, not to
issue commands like "make me a wolf". The character preset is the whole-person
substitution wording from Decart's prompt guide, because a preset that asks for
the face to be kept consistent is carried out as exactly that: the model keeps
the source body and the source clothes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .prompting import CHARACTER_TEMPLATE


@dataclass(frozen=True)
class Preset:
    key: str
    label: str
    prompt: str
    needs_reference: bool = False


PRESETS: tuple[Preset, ...] = (
    Preset(
        key="character",
        label="Character swap (use reference image)",
        prompt=CHARACTER_TEMPLATE,
        needs_reference=True,
    ),
    Preset(
        key="identity",
        label="Same person, studio lighting",
        prompt=(
            "Keep the person exactly as they are but relight them like a studio "
            "portrait: soft key light from the left, gentle fill, clean neutral "
            "background, sharp focus on the eyes."
        ),
    ),
    Preset(
        key="anime",
        label="Anime character",
        prompt=(
            "Render the person as a 2D anime character with clean cel shading and "
            "large expressive eyes, preserving their exact pose, motion and "
            "framing. Keep the background as a soft hand-painted anime interior."
        ),
    ),
    Preset(
        key="cinematic",
        label="Cinematic film look",
        prompt=(
            "Keep the person and the room unchanged but grade the image like a "
            "cinematic film: warm highlights, teal shadows, shallow depth of "
            "field, subtle film grain."
        ),
    ),
    Preset(
        key="suit",
        label="Formal outfit",
        prompt=(
            "Dress the person in a well-fitted dark tailored suit with a crisp "
            "white shirt, keeping their face, hair and body motion unchanged."
        ),
    ),
    Preset(
        key="custom",
        label="Custom prompt",
        prompt="",
    ),
)

BY_KEY = {p.key: p for p in PRESETS}
DEFAULT_KEY = "character"

# The character preset shipped with this wording until 0.1.1. It asks for the
# face, hair and clothing to be held consistent, and Decart's model does exactly
# that: the face changes and the wearer's body and clothes stay. Anyone who has
# launched the app since has it saved, so it has to be migrated rather than
# merged into the new wording.
LEGACY_CHARACTER_PROMPT = (
    "Substitute the character in the video with the person in the reference "
    "image. Keep their face, hair, skin tone, body shape and clothing "
    "consistent in every frame, with natural head movement and lighting "
    "that matches the room."
)


def migrate(text: str) -> str:
    """Replace a saved face-locking character prompt with the whole-person one."""
    if (text or "").strip() == LEGACY_CHARACTER_PROMPT:
        return CHARACTER_TEMPLATE
    return text


def get(key: str) -> Preset:
    return BY_KEY.get(key, BY_KEY[DEFAULT_KEY])


def system_prompt(preset_key: str, custom: str) -> str:
    """The prompt actually sent to Lucy: preset text plus the user's additions."""
    preset = get(preset_key)
    text = preset.prompt.strip()
    extra = (custom or "").strip()
    if preset.key == "custom":
        return extra
    if extra and extra != text:
        return f"{text}\n\nAdditional direction: {extra}"
    return text
