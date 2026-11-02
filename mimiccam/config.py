"""Persistent settings and secret storage.

The Decart API key never goes into settings.json. It lives in the Windows
Credential Manager via ``keyring`` (or in the ``DECART_API_KEY`` environment
variable for CI / power users).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

APP_NAME = "MimicCam"
ENV_API_KEY = "DECART_API_KEY"
_KEYRING_SERVICE = APP_NAME
_KEYRING_USER = "decart_api_key"


def app_data_dir() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".config"
    path = root / APP_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def characters_dir() -> Path:
    path = app_data_dir() / "characters"
    path.mkdir(parents=True, exist_ok=True)
    return path


SETTINGS_PATH = app_data_dir() / "settings.json"


@dataclass
class Settings:
    """Everything the app remembers between launches (non-secret)."""

    camera_index: int = 0
    camera_name: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 30
    mirror: bool = True
    mode: str = "passthrough"  # "passthrough" | "decart"
    model: str = "lucy-2.5"
    prompt: str = ""
    preset: str = "character"
    reference_image: str = ""
    # What the character wears, in the user's own words. Lucy only sees the
    # clothes inside the photo, so a chest-up reference needs this to cover the
    # rest of the body.
    character_notes: str = ""
    studio_background: bool = False
    # Try-on mode: Lucy replaces the outfit from a garment photo instead of
    # swapping the person. ``prompt`` then describes the garment.
    tryon: bool = False
    garment_image: str = ""
    # Paste the live mouth back over the swapped face. The swap network has no
    # mouth tracking, so without this the character's lips stop matching what
    # is being said.
    keep_mouth: bool = True
    fast_mode: bool = False
    # Draw the raw camera as a small box over the output preview.
    pip_camera: bool = True
    # Decart has no balance API, so the figure is typed in from the dashboard
    # once and MimicCam subtracts the seconds it metered itself.
    balance_usd: float = 0.0
    balance_spent: float = 0.0
    # Stop paying for Lucy while nothing moves in front of the camera.
    pause_when_idle: bool = True
    vcam_backend: str = ""  # "" = auto-detect
    window_width: int = 1180
    window_height: int = 760

    @classmethod
    def load(cls) -> "Settings":
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        tmp = SETTINGS_PATH.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
            tmp.replace(SETTINGS_PATH)
        except OSError:
            # Settings are a convenience; never crash the app over them.
            pass


def get_api_key() -> str:
    """Resolve the Decart key: env var first, then the credential store."""
    env = os.environ.get(ENV_API_KEY, "").strip()
    if env:
        return env
    try:
        import keyring

        return (keyring.get_password(_KEYRING_SERVICE, _KEYRING_USER) or "").strip()
    except Exception:
        return ""


def set_api_key(value: str) -> bool:
    """Store the key in the OS credential store. Returns True on success."""
    value = (value or "").strip()
    try:
        import keyring

        if value:
            keyring.set_password(_KEYRING_SERVICE, _KEYRING_USER, value)
        else:
            try:
                keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USER)
            except Exception:
                pass
        return True
    except Exception:
        return False


def mask_key(key: str) -> str:
    if not key:
        return "(none)"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"
