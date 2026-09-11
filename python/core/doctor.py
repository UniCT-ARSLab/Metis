"""Runtime diagnostics used by the Metis CLI and Godot editor add-on."""

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from importlib import metadata
from pathlib import Path


def _ensure_nvidia_pip_libs_on_path():
    # TensorFlow only finds the pip-installed CUDA libraries (nvidia-*-cu12) when their lib dirs are
    # on LD_LIBRARY_PATH BEFORE it is imported. The training entry points (run.py / dqn.py / ppo.py /
    # common.py) do exactly this and re-exec; the doctor is invoked directly by the editor add-on, so
    # without the same bootstrap it reports "no GPU" even on machines whose training runs use the GPU
    # (that mismatch made the CUDA-profile validation fail). Guard prevents an exec loop.
    if os.environ.get("GODOT_GYM_TF_LD_READY") == "1" or platform.system() != "Linux":
        return
    purelib = Path(sysconfig.get_paths()["purelib"])
    nvidia_dir = purelib / "nvidia"
    if not nvidia_dir.exists():
        return
    lib_dirs = [
        str(package_dir / "lib")
        for package_dir in nvidia_dir.iterdir()
        if (package_dir / "lib").is_dir()
    ]
    if not lib_dirs:
        return
    current_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    missing_paths = [p for p in lib_dirs if p not in current_paths]
    os.environ["GODOT_GYM_TF_LD_READY"] = "1"
    if missing_paths:
        os.environ["LD_LIBRARY_PATH"] = ":".join(missing_paths + current_paths)
        original_args = list(getattr(sys, "orig_argv", sys.argv))
        os.execv(sys.executable, [sys.executable] + original_args[1:])


_ensure_nvidia_pip_libs_on_path()


PROFILE_MODULES = {
    "core": (
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
    ),
    "native": (
        ("tensorflow", "tensorflow"),
    ),
    "dashboard": (
        ("flask", "flask"),
        ("flask-sock", "flask_sock"),
    ),
    "export": (
        ("tf2onnx", "tf2onnx"),
    ),
    "sb3": (
        ("torch", "torch"),
        ("stable-baselines3", "stable_baselines3"),
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate a Metis Python runtime and its optional profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILE_MODULES),
        default=None,
        help="Runtime profile to validate. May be passed more than once.",
    )
    parser.add_argument("--godot-bin", default=os.environ.get("GODOT_BIN"))
    parser.add_argument(
        "--expected-metis-version",
        default=None,
        help="Fail when the installed metis-rl distribution has another version.",
    )
    parser.add_argument(
        "--require-accelerator",
        choices=["tensorflow"],
        default=None,
        help="Fail when the selected framework cannot see a hardware accelerator.",
    )
    parser.add_argument(
        "--accelerators",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask TensorFlow or PyTorch for visible accelerators when installed.",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser.parse_args(argv)


def _distribution_version(distribution_name):
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None


def _module_status(distribution_name, module_name):
    version = _distribution_version(distribution_name)
    available = importlib.util.find_spec(module_name) is not None
    return {
        "available": available,
        "version": version,
        "module": module_name,
    }


def _godot_status(executable):
    if not executable:
        return {"available": False, "path": None, "version": None}
    resolved = shutil.which(executable) or (
        str(Path(executable).expanduser().resolve())
        if Path(executable).expanduser().is_file()
        else None
    )
    if resolved is None:
        return {"available": False, "path": executable, "version": None}
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        version = (completed.stdout or completed.stderr).strip() or None
        available = completed.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        available = False
        version = None
    return {"available": available, "path": resolved, "version": version}


def _accelerator_status(packages):
    result = {}
    if packages.get("tensorflow", {}).get("available"):
        try:
            tensorflow = importlib.import_module("tensorflow")
            result["tensorflow"] = [
                device.name
                for device in tensorflow.config.list_physical_devices("GPU")
            ]
        except Exception as exc:
            result["tensorflow_error"] = str(exc)
    if packages.get("torch", {}).get("available"):
        try:
            torch = importlib.import_module("torch")
            result["torch"] = {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "mps_available": bool(
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                ),
            }
        except Exception as exc:
            result["torch_error"] = str(exc)
    return result


def build_report(
    profiles,
    godot_bin=None,
    accelerators=True,
    expected_metis_version=None,
    require_accelerator=None,
):
    requested_profiles = list(dict.fromkeys(["core", *profiles]))
    requested_modules = {}
    for profile in requested_profiles:
        for distribution_name, module_name in PROFILE_MODULES[profile]:
            requested_modules[distribution_name] = module_name

    packages = {
        distribution_name: _module_status(distribution_name, module_name)
        for distribution_name, module_name in requested_modules.items()
    }
    runtime_version = _distribution_version("metis-rl")
    report = {
        "status": "ok",
        "metis_version": runtime_version or "source",
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "profiles": requested_profiles,
        "packages": packages,
        "godot": _godot_status(godot_bin),
        "accelerators": _accelerator_status(packages) if accelerators else {},
    }
    missing = sorted(
        name for name, status in packages.items() if not status["available"]
    )
    if missing:
        report["status"] = "error"
        report["missing"] = missing
    if (
        expected_metis_version is not None
        and runtime_version != expected_metis_version
    ):
        report["status"] = "error"
        report["version_mismatch"] = {
            "expected": expected_metis_version,
            "installed": runtime_version,
        }
    if require_accelerator == "tensorflow":
        tensorflow_devices = report["accelerators"].get("tensorflow", [])
        if not tensorflow_devices:
            report["status"] = "error"
            report["accelerator_error"] = (
                "TensorFlow cannot see a GPU for the selected runtime profile."
            )
    return report


def _print_report(report):
    print(
        f"Metis runtime: {report['status']} version={report['metis_version']} "
        f"python={report['python']['version']}"
    )
    print(f"Python executable: {report['python']['executable']}")
    for name, status in report["packages"].items():
        label = status["version"] or "not installed"
        print(f"  {name:<20} {label}")
    godot = report["godot"]
    if godot["path"]:
        print(
            f"Godot: {'ok' if godot['available'] else 'unavailable'} "
            f"{godot['version'] or godot['path']}"
        )
    if report["accelerators"]:
        print(f"Accelerators: {report['accelerators']}")
    if report.get("missing"):
        print(f"Missing packages: {', '.join(report['missing'])}")
    if report.get("version_mismatch"):
        mismatch = report["version_mismatch"]
        print(
            "Metis version mismatch: "
            f"expected {mismatch['expected']}, installed {mismatch['installed']}"
        )
    if report.get("accelerator_error"):
        print(f"Accelerator error: {report['accelerator_error']}")


def main(argv=None):
    args = parse_args(argv)
    profiles = args.profile or ["core"]
    report = build_report(
        profiles,
        godot_bin=args.godot_bin,
        accelerators=args.accelerators,
        expected_metis_version=args.expected_metis_version,
        require_accelerator=args.require_accelerator,
    )
    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_report(report)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
