"""Frozen entry point for PyInstaller.

PyInstaller cannot execute ``mimiccam/__main__.py`` directly because relative
imports need the package context, so this thin launcher imports the package.

A windowed frozen app has no console, so an unhandled exception would otherwise
surface as PyInstaller's terse "Failed to execute script" dialog with no clue
about what went wrong. Everything is caught here, written to a log the user can
send us, and shown in a dialog that names the actual problem.
"""

import sys
import traceback
from pathlib import Path


def _crash_log_path() -> Path:
    base = Path.home() / "AppData" / "Roaming"
    folder = base / "MimicCam"
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError:
        folder = Path.cwd()
    return folder / "crash.log"


def _report(exc: BaseException, interactive: bool) -> None:
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    log = _crash_log_path()
    try:
        log.write_text(details, encoding="utf-8")
    except OSError:
        log = Path("crash.log")

    message = f"{type(exc).__name__}: {exc}\n\nFull details were written to:\n{log}"
    if "too long" in str(exc).lower():
        message += (
            "\n\nMimicCam's folder path is too deep for Windows to load its "
            "virtual-camera driver. Move the folder somewhere short, such as "
            r"C:\MimicCam, and start it again."
        )
    if not interactive:
        # A CLI run (--list-backends and friends) must exit, not wait for a click.
        print(message, file=sys.stderr)
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "MimicCam could not start", 0x10)
    except Exception:
        print(message)


if __name__ == "__main__":
    try:
        from mimiccam.__main__ import main

        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - a frozen app must never die silently
        _report(exc, interactive=len(sys.argv) <= 1)
        sys.exit(1)
