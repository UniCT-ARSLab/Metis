#!/usr/bin/env python3
"""Create or validate a project-local Metis Python runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


ACTIVE_PROCESS = None
MINIMUM_PYTHON = (3, 11)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["cpu", "linux_cuda", "macos_metal"],
        required=True,
    )
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--export", action="store_true", dest="export_tools")
    parser.add_argument("--sb3", action="store_true")
    parser.add_argument("--existing", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--godot-bin", default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    return parser.parse_args()


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_status(args, state, step, progress, message, **extra):
    payload = {
        "format": "metis-runtime-setup",
        "format_version": 1,
        "state": state,
        "step": step,
        "progress": float(progress),
        "message": message,
    }
    payload.update(extra)
    write_json_atomic(args.status_path, payload)


def runtime_python(venv_dir):
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_logged(command, log, cwd):
    global ACTIVE_PROCESS
    log.write(f"$ {' '.join(map(str, command))}\n")
    log.flush()
    ACTIVE_PROCESS = subprocess.Popen(
        [str(value) for value in command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in ACTIVE_PROCESS.stdout:
        log.write(line)
        log.flush()
    return_code = ACTIVE_PROCESS.wait()
    ACTIVE_PROCESS = None
    if return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {' '.join(map(str, command))}"
        )


def run_logged_capture(command, log, cwd):
    """Like run_logged but never raises: returns (return_code, parsed_json_report_or_None).

    Used to validate a pre-existing environment, where a non-zero doctor exit is informational
    (missing optional components / no accelerator) rather than a fatal setup failure.
    """
    global ACTIVE_PROCESS
    log.write(f"$ {' '.join(map(str, command))}\n")
    log.flush()
    ACTIVE_PROCESS = subprocess.Popen(
        [str(value) for value in command],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured = []
    for line in ACTIVE_PROCESS.stdout:
        log.write(line)
        log.flush()
        captured.append(line)
    return_code = ACTIVE_PROCESS.wait()
    ACTIVE_PROCESS = None
    report = None
    text = "".join(captured)
    brace = text.find("{")
    if brace != -1:
        try:
            report = json.loads(text[brace:])
        except json.JSONDecodeError:
            report = None
    return return_code, report


def stop_active_process(_signal_number, _frame):
    if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
        ACTIVE_PROCESS.terminate()
    raise KeyboardInterrupt


def load_manifest(payload_dir, required=True):
    manifest_path = payload_dir / "runtime_manifest.json"
    if not manifest_path.is_file():
        if not required:
            return {
                "plugin_version": "development",
                "python_version": "development",
            }
        raise RuntimeError(
            "The add-on has no packaged Python runtime. Build a Metis release or "
            "select an existing environment."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_bootstrap(profile, version_info=None, system=None, machine=None):
    version_info = version_info or sys.version_info
    if tuple(version_info[:2]) < MINIMUM_PYTHON:
        required = ".".join(map(str, MINIMUM_PYTHON))
        actual = ".".join(map(str, version_info[:3]))
        raise RuntimeError(
            f"Metis requires Python {required} or newer; selected interpreter is "
            f"Python {actual}."
        )

    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if profile == "linux_cuda" and system != "Linux":
        raise RuntimeError("The NVIDIA CUDA profile is available only on Linux.")
    if profile == "macos_metal" and (
        system != "Darwin" or machine not in {"arm64", "aarch64"}
    ):
        raise RuntimeError(
            "The Apple Metal profile requires macOS on Apple Silicon."
        )


def validate_payload(payload_dir, manifest):
    required_keys = {
        "plugin_version",
        "python_version",
        "wheel",
        "wheel_sha256",
        "profiles",
        "extras",
        "requirements",
    }
    missing_keys = sorted(required_keys - set(manifest))
    if missing_keys:
        raise RuntimeError(
            f"The packaged runtime manifest is incomplete: {missing_keys}"
        )
    if manifest["plugin_version"] != manifest["python_version"]:
        raise RuntimeError(
            "The Godot add-on and Python runtime versions do not match."
        )
    if not isinstance(manifest["profiles"], dict) or not manifest["profiles"]:
        raise RuntimeError("The packaged runtime has no installation profiles.")
    if not isinstance(manifest["extras"], dict):
        raise RuntimeError("The packaged runtime has invalid optional components.")
    if not isinstance(manifest["requirements"], dict):
        raise RuntimeError("The packaged runtime has invalid requirement checksums.")

    wheel_path = payload_dir / manifest["wheel"]
    if not wheel_path.is_file():
        raise RuntimeError(f"The packaged Metis wheel is missing: {wheel_path}")
    if sha256(wheel_path) != manifest["wheel_sha256"]:
        raise RuntimeError(f"The packaged Metis wheel checksum is invalid: {wheel_path}")

    for filename, expected_hash in manifest["requirements"].items():
        requirement_path = payload_dir / filename
        if not requirement_path.is_file():
            raise RuntimeError(
                f"The packaged requirement profile is missing: {requirement_path}"
            )
        if sha256(requirement_path) != expected_hash:
            raise RuntimeError(
                f"The requirement profile checksum is invalid: {requirement_path}"
            )
    for group_name in ("profiles", "extras"):
        for component, filename in manifest[group_name].items():
            if filename not in manifest["requirements"]:
                raise RuntimeError(
                    f"The {component!r} {group_name} entry references an unchecked "
                    f"requirement file: {filename}"
                )


def validate_existing_metis(manifest):
    expected = manifest.get("python_version")
    if expected in {None, "development"}:
        return
    try:
        installed = metadata.version("metis-rl")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "The selected Python environment does not contain metis-rl."
        ) from exc
    if installed != expected:
        raise RuntimeError(
            f"The selected environment contains metis-rl {installed}, but the "
            f"Godot add-on requires {expected}."
        )


def install_managed_runtime(args, log, manifest):
    metis_dir = args.project_dir / ".metis"
    venv_dir = metis_dir / "venv"
    if args.recreate and venv_dir.exists():
        write_status(args, "running", "remove_runtime", 0.05, "Removing the previous managed runtime")
        shutil.rmtree(venv_dir)

    if not runtime_python(venv_dir).is_file():
        write_status(args, "running", "create_venv", 0.10, "Creating the project virtual environment")
        run_logged(
            [sys.executable, "-m", "venv", str(venv_dir)],
            log,
            args.project_dir,
        )

    python_executable = runtime_python(venv_dir)
    profile_requirement = manifest["profiles"].get(args.profile)
    if profile_requirement is None:
        raise RuntimeError(
            f"The release does not provide the {args.profile!r} runtime profile."
        )
    requirement_paths = [args.payload_dir / profile_requirement]
    if args.dashboard:
        requirement_paths.append(
            args.payload_dir / manifest["extras"]["dashboard"])
    if args.export_tools:
        requirement_paths.append(
            args.payload_dir / manifest["extras"]["export"])
    if args.sb3:
        requirement_paths.append(
            args.payload_dir / manifest["extras"]["sb3"])

    for index, requirement_path in enumerate(requirement_paths):
        progress = 0.25 + (0.35 * index / max(len(requirement_paths), 1))
        write_status(
            args,
            "running",
            "install_requirements",
            progress,
            f"Installing {requirement_path.name}",
        )
        run_logged(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                requirement_path,
            ],
            log,
            args.project_dir,
        )

    wheel_path = args.payload_dir / manifest["wheel"]
    if not wheel_path.is_file():
        raise RuntimeError(f"The packaged Metis wheel is missing: {wheel_path}")
    write_status(args, "running", "install_metis", 0.70, "Installing the matching Metis wheel")
    run_logged(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            "--upgrade",
            wheel_path,
        ],
        log,
        args.project_dir,
    )
    return python_executable, manifest


def validate_runtime(args, log, python_executable, manifest):
    doctor_profiles = ["core", "native"]
    if args.dashboard:
        doctor_profiles.append("dashboard")
    if args.export_tools:
        doctor_profiles.append("export")
    if args.sb3:
        doctor_profiles.append("sb3")
    if args.source_root is not None:
        doctor_entrypoint = args.source_root.resolve() / "core" / "doctor.py"
        command = [python_executable, doctor_entrypoint, "--json"]
    else:
        command = [python_executable, "-m", "core.doctor", "--json"]
    for profile in doctor_profiles:
        command.extend(["--profile", profile])
    if args.godot_bin:
        command.extend(["--godot-bin", args.godot_bin])
    expected_version = manifest.get("python_version")
    if args.source_root is None and expected_version not in {None, "development"}:
        command.extend(["--expected-metis-version", str(expected_version)])
    # Only DEMAND a working accelerator for a MANAGED GPU install. When validating an EXISTING
    # environment the user may knowingly run on CPU, or their TensorFlow build simply cannot see the
    # GPU inside this subprocess (missing CUDA libs on the path) -- that must not hard-fail validation.
    if not args.existing and args.profile in {"linux_cuda", "macos_metal"}:
        command.extend(["--require-accelerator", "tensorflow"])

    write_status(args, "running", "validate", 0.88, "Validating the installed runtime")
    if args.existing:
        # Validating a pre-existing/source environment: the doctor is INFORMATIONAL, not a gate.
        # Missing OPTIONAL components (dashboard/export/sb3) or an unavailable accelerator are
        # surfaced as warnings; we only fail if the CORE runtime is absent.
        return_code, report = run_logged_capture(command, log, args.project_dir)
        packages = (report or {}).get("packages", {})
        essential = ("tensorflow", "gymnasium", "numpy")
        missing_core = [
            name for name in essential
            if not packages.get(name, {}).get("available", False)
        ]
        if report is None or missing_core:
            detail = (
                "the environment is missing core packages: " + ", ".join(missing_core)
                if missing_core
                else f"the runtime check exited with code {return_code}"
            )
            raise RuntimeError(f"Cannot validate the selected environment ({detail}).")
        warnings = list((report or {}).get("missing", []))
        if (report or {}).get("accelerator_error"):
            warnings.append(str(report["accelerator_error"]))
        if warnings:
            log.write(
                "NOTE: core runtime is ready; optional components unavailable: "
                + "; ".join(warnings)
                + "\n"
            )
            log.flush()
    else:
        run_logged(command, log, args.project_dir)
    runtime_config = {
        "format": "metis-project-runtime",
        "format_version": 1,
        "managed": not args.existing,
        "python": str(python_executable),
        "profile": args.profile,
        "dashboard": args.dashboard,
        "export": args.export_tools,
        "sb3": args.sb3,
        "plugin_version": manifest.get("plugin_version"),
        "python_version": manifest.get("python_version"),
        "python_runtime_version": platform.python_version(),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
    }
    if args.godot_bin:
        runtime_config["godot_bin"] = str(args.godot_bin)
    if args.source_root is not None:
        runtime_config["source_root"] = str(args.source_root.resolve())
    config_path = args.project_dir / ".metis" / "runtime.json"
    write_json_atomic(config_path, runtime_config)
    return runtime_config


def main():
    args = parse_args()
    args.project_dir = args.project_dir.resolve()
    args.payload_dir = args.payload_dir.resolve()
    args.status_path = args.status_path.resolve()
    args.log_path = args.log_path.resolve()
    args.log_path.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, stop_active_process)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_active_process)

    try:
        with args.log_path.open("w", encoding="utf-8") as log:
            write_status(args, "running", "prepare", 0.01, "Preparing the Metis runtime")
            validate_bootstrap(args.profile)
            manifest = load_manifest(
                args.payload_dir,
                required=not args.existing,
            )
            if manifest.get("python_version") != "development":
                validate_payload(args.payload_dir, manifest)
            if args.existing:
                if args.source_root is None:
                    validate_existing_metis(manifest)
                python_executable = Path(sys.executable)
            else:
                python_executable, manifest = install_managed_runtime(
                    args,
                    log,
                    manifest,
                )
            runtime_config = validate_runtime(
                args,
                log,
                python_executable,
                manifest,
            )
        write_status(
            args,
            "ready",
            "complete",
            1.0,
            "Metis runtime is ready",
            runtime=runtime_config,
        )
        return 0
    except KeyboardInterrupt:
        write_status(args, "cancelled", "cancelled", 0.0, "Runtime setup was cancelled")
        return 130
    except Exception as exc:
        with args.log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        write_status(
            args,
            "error",
            "failed",
            0.0,
            str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
