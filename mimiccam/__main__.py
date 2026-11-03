"""Entry point: ``python -m mimiccam`` starts the GUI.

Also provides the small CLI used by the test suite and by support requests:

    python -m mimiccam --list-cameras
    python -m mimiccam --list-backends
    python -m mimiccam --selftest 15 --camera 1
    python -m mimiccam --selftest 10 --mode local --reference me.jpg
    python -m mimiccam --selftest 10 --mode local --reference me.jpg --studio
    python -m mimiccam --check-lucy --reference me.jpg --notes "black suit"
"""

from __future__ import annotations

import argparse
import sys
import time

from . import __copyright__, __url__, __version__


def _list_cameras() -> int:
    from .capture import probe_cameras

    entries = probe_cameras(6)
    if not entries:
        print("No working camera found.")
        return 1
    for entry in entries:
        print(f"index={entry.index} {entry.width}x{entry.height} luma={entry.mean_luma:.0f}")
    return 0


def _list_backends() -> int:
    from .vcam import available_backends

    backends = available_backends()
    print("Available virtual-camera backends:", ", ".join(backends) if backends else "none")
    return 0 if backends else 1


def _check_local() -> int:
    """Load the offline models and run one swap: the build's ONNX smoke test.

    Exercising the whole onnxruntime -> DirectML path on a synthetic frame keeps
    the check independent of cameras, faces and people.
    """
    import numpy as np

    from . import face, gpu, models

    absent = models.missing()
    if absent:
        print("[local] models not installed - skipping:", models.describe_missing(absent))
        return 0

    try:
        started = time.perf_counter()
        detector = face.FaceDetector(models.path("detector"))
        embedder = face.FaceEmbedder(models.path("identity"))
        swapper = face.FaceSwapper(models.path("swapper"))
        provider = swapper.session.get_providers()[0]
        adapter = gpu.nvidia_adapter()
        where = gpu.adapter_name(face.default_providers())

        frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        landmarks = np.array(
            [[250, 140], [390, 140], [320, 190], [270, 250], [370, 250]], np.float32
        )
        rng = np.random.default_rng(0)
        embedding = rng.standard_normal(512).astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        texture = swapper.render(frame, landmarks, embedding)
        output = swapper.swap(frame, landmarks, embedding)
    except Exception as exc:  # noqa: BLE001
        print(f"[local] FAIL - {exc}")
        return 1

    if texture.shape != (128, 128, 3) or texture.dtype != np.uint8:
        print(f"[local] FAIL - unexpected render {texture.shape} {texture.dtype}")
        return 1
    if float(texture.std()) < 1.0:
        print("[local] FAIL - the swap network returned a constant image")
        return 1

    # Measure inside the face box only: the paste is a few percent of the frame,
    # so a whole-frame mean would call a working swap "barely changed".
    patch = (slice(110, 300), slice(220, 420))
    changed = float(
        np.abs(output[patch].astype(int) - frame[patch].astype(int)).mean()
    )
    elapsed = (time.perf_counter() - started) * 1000
    print(f"[local] provider={provider} adapter={where} "
          f"load+swap={elapsed:.0f}ms face_change={changed:.1f}")
    print(f"[local] render={texture.shape} std={float(texture.std()):.1f} "
          f"detect={len(detector.detect(frame))}")
    if changed < 0.2:
        print("[local] FAIL - the swap did not modify the frame")
        return 1
    if adapter is not None and provider != "DmlExecutionProvider":
        print(
            "[local] FAIL - an NVIDIA adapter is present but the models did not run "
            f"on it (provider is {provider})"
        )
        return 1
    if provider != "DmlExecutionProvider":
        print("[local] WARNING - no NVIDIA GPU here; the models ran on the CPU")
    print("[local] PASS")
    return 0


def _check_lucy(reference_path: str, notes: str) -> int:
    """Show what Lucy would actually be sent, without opening a billed session.

    Exists because the sentence decides whether the swap is a whole person or a
    face paste, and because a frozen build proves in one line that the reference
    and prompt modules are bundled.
    """
    from . import prompting, reference

    if not reference_path:
        print("[lucy] FAIL - pass --reference <character photo>")
        return 1
    try:
        appearance = reference.analyse(reference_path)
        data = reference.prepare_reference(reference_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[lucy] FAIL - {exc}")
        return 1

    prompt = prompting.character_prompt(
        prompting.CHARACTER_TEMPLATE, appearance=appearance, notes=notes
    )
    framing = appearance.framing
    print(
        f"[lucy] reference={len(data)} bytes framing={framing.kind} "
        f"face={framing.face_fraction:.2f} of the photo",
        flush=True,
    )
    print(
        f"[lucy] hair={appearance.hair or '-'} "
        f"garments={','.join(appearance.garments) or '-'}",
        flush=True,
    )
    print(f"[lucy] prompt ({len(prompt)} chars): {prompt}", flush=True)

    ok = (
        0 < len(prompt) <= prompting.MAX_PROMPT_CHARS
        and "head to toe" in prompt
        and "keep their face, hair" not in prompt.lower()
    )
    print("[lucy] PASS" if ok else "[lucy] FAIL - the prompt is not a whole-person swap")
    return 0 if ok else 1


def _selftest(seconds: float, camera: int, mode: str = "passthrough",
              reference: str = "", studio: bool = False) -> int:
    """Camera -> virtual camera with no UI, used by the automated verification."""
    from .pipeline import Pipeline

    def log(message: str) -> None:
        print(f"[app] {message}", flush=True)

    engine = None
    if mode == "local":
        from .local_engine import LocalEngine

        try:
            engine = LocalEngine(
                reference_image=reference or None,
                width=1280,
                height=720,
                fps=30,
                studio_background=studio and bool(reference),
                studio_image=reference if studio else None,
                on_log=log,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[result] FAIL - local engine unavailable: {exc}", flush=True)
            return 1

    pipeline = Pipeline(
        camera_index=camera,
        width=1280,
        height=720,
        fps=30,
        mirror=True,
        on_log=log,
        on_preview=lambda _frame: None,
        on_stats=lambda stats: print(f"[stats] {stats}", flush=True),
    )
    if not pipeline.start(engine):
        print("[result] FAIL - pipeline did not start", flush=True)
        return 1

    print("[app] READY", flush=True)
    try:
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(0.5)
    finally:
        sent = pipeline.vcam.frames_sent
        pipeline.stop()

    ok = sent > 0
    print(f"[result] frames_sent={sent} held={pipeline.held_frames}", flush=True)
    print("[result] PASS" if ok else "[result] FAIL - no frames reached the virtual camera", flush=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mimiccam", description="MimicCam")
    # Says who made it, in the one place a command-line user looks first.
    parser.add_argument(
        "--version", action="version",
        version=f"MimicCam {__version__} - {__copyright__} - GPL-2.0-or-later\n{__url__}",
        help="print the version, the copyright and the repository, then exit",
    )
    parser.add_argument("--list-cameras", action="store_true", help="probe cameras and exit")
    parser.add_argument("--list-backends", action="store_true", help="probe virtual cameras and exit")
    parser.add_argument(
        "--check-local", action="store_true",
        help="load the offline models and run one swap, then exit",
    )
    parser.add_argument(
        "--check-lucy", action="store_true",
        help="print the prompt Lucy would receive for --reference, then exit",
    )
    parser.add_argument(
        "--notes", default="", help="with --check-lucy: the character's look notes",
    )
    parser.add_argument("--selftest", type=float, metavar="SECONDS", help="run headless")
    parser.add_argument("--camera", type=int, default=0, help="camera index for --selftest")
    parser.add_argument(
        "--mode", choices=("passthrough", "local"), default="passthrough",
        help="engine for --selftest",
    )
    parser.add_argument(
        "--reference", default="", help="character photo for --mode local",
    )
    parser.add_argument(
        "--studio", action="store_true",
        help="with --mode local: seat you in the reference photo's room",
    )
    args = parser.parse_args(argv)

    if args.list_cameras:
        return _list_cameras()
    if args.list_backends:
        return _list_backends()
    if args.check_local:
        return _check_local()
    if args.check_lucy:
        return _check_lucy(args.reference, args.notes)
    if args.selftest:
        return _selftest(args.selftest, args.camera, args.mode, args.reference, args.studio)

    from .ui import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
