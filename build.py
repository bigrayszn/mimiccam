"""Build MimicCam for Windows.

    python build.py                # portable folder + zip
    python build.py --installer    # also run Inno Setup if it is installed

Output:
    dist/MimicCam/MimicCam.exe
    dist/MimicCam-<version>-win64-portable.zip
    dist/MimicCam-<version>-setup.exe      (with --installer)
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
ICON = ROOT / "assets" / "mimiccam.ico"

# Windows refuses to load a DLL when the full path nears MAX_PATH. The deepest
# bundled name is pyvirtualcam's native module, roughly this long.
NATIVE_NAME_LEN = len("_internal\\pyvirtualcam\\_native_windows_unity_capture.cp310-win_amd64.pyd")
PATH_BUDGET = 240

sys.path.insert(0, str(ROOT))
from mimiccam import __version__  # noqa: E402


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def ensure_pyinstaller() -> None:
    if importlib.util.find_spec("PyInstaller") is None:
        run([sys.executable, "-m", "pip", "install", "pyinstaller>=6.6"])


def pyinstaller_args() -> list[str]:
    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean",
        "--windowed",
        "--name", "MimicCam",
        "--onedir",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        "--specpath", str(BUILD),
        # The RTC/Decart stack loads native modules and plugins dynamically.
        "--collect-all", "livekit",
        "--collect-all", "decart",
        "--collect-all", "av",
        "--hidden-import", "keyring.backends.Windows",
        "--hidden-import", "pyvirtualcam",
        # Offline engine: onnxruntime is imported lazily and ships DirectML.dll
        # next to its own package, so both have to be collected explicitly.
        "--collect-all", "onnxruntime",
        "--hidden-import", "onnx",
        # Keep the bundle sane: we never use these Qt modules.
        "--exclude-module", "PySide6.QtWebEngineCore",
        "--exclude-module", "PySide6.QtWebEngineWidgets",
        "--exclude-module", "PySide6.QtQuick3D",
        "--exclude-module", "PySide6.Qt3DCore",
        "--exclude-module", "tkinter",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pytest",
    ]
    if ICON.exists():
        args += ["--icon", str(ICON)]
    args.append(str(ROOT / "main.py"))
    return args


def net(path: Path) -> str:
    """Backslash form Windows actually sees, for length checks and logs."""
    return str(path).replace("/", "\\")


def path_budget_note(exe: Path) -> None:
    deepest = len(net(exe)) + NATIVE_NAME_LEN
    if deepest + 12 > PATH_BUDGET:
        print(
            "\nWARNING: this build lives at\n  "
            f"{net(exe)}\n"
            f"The deepest bundled file is {deepest} characters, close to (or past) Windows' "
            f"{PATH_BUDGET}-character practical limit. Users who unzip this here will not be "
            "able to load the virtual-camera driver. Build into a short path instead:\n"
            "  python build.py --dist C:\\MimicCam-dist"
        )
    else:
        print(f"\nPath budget: deepest bundled file is {deepest} characters (limit {PATH_BUDGET}).")


def smoke_test(exe: Path) -> None:
    """Run the frozen binary before publishing it.

    A build can succeed and still produce an exe that cannot import its own native
    dependencies; this catches that here rather than in a user's call.
    """
    print(f"\nSmoke test: {exe.name} --list-backends")
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(
                [str(exe), "--list-backends"], capture_output=True, text=True, timeout=180
            )
        except subprocess.TimeoutExpired:
            raise SystemExit(
                "SMOKE TEST FAILED: the frozen exe did not exit. This usually means an "
                "error dialog is waiting behind the scenes - run it by hand to see it."
            )
        if proc.returncode == 0 or attempt == attempts:
            break
        # The OBS virtual camera reports "no backend" for a second or two while its
        # filter is starting, which is a machine state, not a broken build. A build
        # that really cannot load its stack fails all three attempts.
        print(f"  attempt {attempt}: no backend yet, retrying in 3s")
        time.sleep(3)
    output = (proc.stdout or "") + (proc.stderr or "")
    print("  exit:", proc.returncode)
    for line in output.strip().splitlines()[-6:]:
        print("  exe|", line)
    if proc.returncode != 0:
        hint = ""
        if "too long" in output.lower():
            hint = (
                "\n  Cause: the build path is too deep for Windows to load a DLL. "
                "Rebuild with --dist C:\\MimicCam-dist"
            )
        raise SystemExit(f"SMOKE TEST FAILED: the frozen exe exited with {proc.returncode}{hint}")
    print("  PASS - the frozen exe imports its stack and finds a virtual camera")

    print(f"\nSmoke test: {exe.name} --check-local")
    proc = subprocess.run(
        [str(exe), "--check-local"], capture_output=True, text=True, timeout=300
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    print("  exit:", proc.returncode)
    for line in output.strip().splitlines()[-6:]:
        print("  exe|", line)
    if proc.returncode != 0:
        raise SystemExit(
            "SMOKE TEST FAILED: the frozen exe could not run the offline engine. "
            "Check that onnxruntime and DirectML.dll were bundled."
        )
    print("  PASS - the frozen exe loads the local models and swaps a frame")


def build_portable() -> Path:
    ensure_pyinstaller()
    if (DIST / "MimicCam").exists():
        shutil.rmtree(DIST / "MimicCam")
    run(pyinstaller_args())

    exe = DIST / "MimicCam" / "MimicCam.exe"
    if not exe.exists():
        raise SystemExit(f"Build finished without producing {exe}")

    path_budget_note(exe)
    smoke_test(exe)

    archive = DIST / f"MimicCam-{__version__}-win64-portable"
    if Path(f"{archive}.zip").exists():
        Path(f"{archive}.zip").unlink()
    shutil.make_archive(str(archive), "zip", str(DIST), "MimicCam")
    size_mb = Path(f"{archive}.zip").stat().st_size / (1024 * 1024)
    print(f"\nPortable build: {archive}.zip ({size_mb:.0f} MB)")
    return archive


def find_iscc() -> Path | None:
    for candidate in (
        Path(r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe"),
        Path(r"C:\Program Files\Inno Setup 6\ISCC.exe"),
    ):
        if candidate.exists():
            return candidate
    return shutil.which("iscc") and Path(shutil.which("iscc"))


def build_installer() -> None:
    iscc = find_iscc()
    if not iscc:
        print("Inno Setup not found - skipping installer. Install it from https://jrsoftware.org/isdl.php")
        return
    run([str(iscc), str(ROOT / "installer.iss")])
    print("Installer written to dist/")


def main() -> int:
    global DIST
    parser = argparse.ArgumentParser()
    parser.add_argument("--installer", action="store_true", help="also build the Inno Setup installer")
    parser.add_argument("--skip-portable", action="store_true", help="only build the installer")
    parser.add_argument(
        "--dist",
        default=str(DIST),
        help="output folder; use a short path such as C:\\MimicCam-dist when the "
        "repository lives deep inside your user folder",
    )
    args = parser.parse_args()
    DIST = Path(args.dist).expanduser().resolve()
    DIST.mkdir(parents=True, exist_ok=True)

    if not args.skip_portable:
        build_portable()
    if args.installer:
        build_installer()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
