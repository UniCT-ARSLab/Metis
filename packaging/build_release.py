#!/usr/bin/env python3
"""Build a version-matched Metis wheel and Godot Asset Library archive."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
ADDON_SOURCE = ROOT / "godot" / "addons" / "metis"
DEFAULT_DIST = ROOT / "dist"
REQUIREMENT_FILES = (
    "requirements-base.txt",
    "requirements.txt",
    "requirements-dashboard.txt",
    "requirements-export.txt",
    "requirements-linux-cuda.txt",
    "requirements-macos-metal.txt",
    "requirements-sb3.txt",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the Metis Python wheel and Asset Library ZIP."
    )
    parser.add_argument("--dist-dir", type=Path, default=DEFAULT_DIST)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter whose build tooling should create the wheel.",
    )
    return parser.parse_args(argv)


def project_versions():
    plugin_config = configparser.ConfigParser()
    plugin_config.read(ADDON_SOURCE / "plugin.cfg", encoding="utf-8")
    plugin_version = plugin_config["plugin"]["version"].strip('"')
    with (PYTHON_SOURCE / "pyproject.toml").open("rb") as stream:
        python_version = tomllib.load(stream)["project"]["version"]
    return plugin_version, python_version


def validate_source():
    plugin_version, python_version = project_versions()
    if plugin_version != python_version:
        raise RuntimeError(
            "Godot and Python versions differ: "
            f"plugin={plugin_version} python={python_version}"
        )
    required = (
        ADDON_SOURCE / "plugin.cfg",
        ADDON_SOURCE / "plugin.gd",
        ADDON_SOURCE / "README.md",
        ADDON_SOURCE / "runtime" / "bridge_server.gd",
        ADDON_SOURCE / "python_setup" / "setup_runtime.py",
        ADDON_SOURCE / "integrations" / "urdf" / "LICENSE",
        ADDON_SOURCE / "integrations" / "stl" / "license.txt",
        ROOT / "LICENSE",
    )
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Release source is incomplete: {', '.join(missing)}")
    return plugin_version


def build_wheel(python_executable, wheel_dir):
    command = [
        python_executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--wheel-dir",
        str(wheel_dir),
        str(PYTHON_SOURCE),
    ]
    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("SOURCE_DATE_EPOCH", "315532800")
    subprocess.run(command, cwd=ROOT, env=environment, check=True)
    wheels = sorted(wheel_dir.glob("metis_rl-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one Metis wheel, found: {wheels}")
    return wheels[0]


def _copy_addon(stage_addon):
    shutil.copytree(
        ADDON_SOURCE,
        stage_addon,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    shutil.copy2(ROOT / "LICENSE", stage_addon / "LICENSE")
    shutil.copy2(ROOT / "godot" / "icon.svg", stage_addon / "logo.svg")


def _write_runtime_payload(stage_addon, wheel, version):
    payload_dir = stage_addon / "python_setup"
    payload_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, payload_dir / wheel.name)
    for filename in REQUIREMENT_FILES:
        shutil.copy2(PYTHON_SOURCE / filename, payload_dir / filename)

    manifest = {
        "format": "metis-runtime",
        "format_version": 1,
        "plugin_version": version,
        "python_distribution": "metis-rl",
        "python_version": version,
        "python_requires": ">=3.11",
        "wheel": wheel.name,
        "wheel_sha256": sha256(wheel),
        "profiles": {
            "cpu": "requirements.txt",
            "linux_cuda": "requirements-linux-cuda.txt",
            "macos_metal": "requirements-macos-metal.txt",
        },
        "extras": {
            "dashboard": "requirements-dashboard.txt",
            "export": "requirements-export.txt",
            "sb3": "requirements-sb3.txt",
        },
        "requirements": {
            filename: sha256(payload_dir / filename)
            for filename in REQUIREMENT_FILES
        },
    }
    (payload_dir / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_deterministic_zip(source_root, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive(archive_path, version):
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        prefix = "addons/metis/"
        required = {
            prefix + "plugin.cfg",
            prefix + "plugin.gd",
            prefix + "README.md",
            prefix + "LICENSE",
            prefix + "logo.svg",
            prefix + "runtime/bridge_server.gd",
            prefix + "python_setup/setup_runtime.py",
            prefix + "python_setup/runtime_manifest.json",
            prefix + "integrations/urdf/LICENSE",
            prefix + "integrations/stl/license.txt",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Asset Library archive is missing: {missing}")
        wheel_names = [
            name
            for name in names
            if name.startswith(prefix + "python_setup/metis_rl-")
            and name.endswith(".whl")
        ]
        if len(wheel_names) != 1 or version not in wheel_names[0]:
            raise RuntimeError(f"Unexpected runtime wheels in archive: {wheel_names}")
        missing_requirements = [
            prefix + "python_setup/" + filename
            for filename in REQUIREMENT_FILES
            if prefix + "python_setup/" + filename not in names
        ]
        if missing_requirements:
            raise RuntimeError(
                "Asset Library archive is missing runtime profiles: "
                f"{missing_requirements}"
            )
        manifest = json.loads(
            archive.read(prefix + "python_setup/runtime_manifest.json")
        )
        wheel_name = prefix + "python_setup/" + manifest["wheel"]
        if hashlib.sha256(archive.read(wheel_name)).hexdigest() != manifest.get(
            "wheel_sha256"
        ):
            raise RuntimeError("The packaged wheel checksum does not match its manifest.")
        for filename, expected_hash in manifest.get("requirements", {}).items():
            payload_name = prefix + "python_setup/" + filename
            if payload_name not in names:
                raise RuntimeError(
                    f"The runtime manifest references a missing file: {payload_name}"
                )
            if hashlib.sha256(archive.read(payload_name)).hexdigest() != expected_hash:
                raise RuntimeError(
                    f"The runtime requirement checksum is invalid: {payload_name}"
                )
        forbidden = [
            name
            for name in names
            if not name.startswith(prefix)
            or "/__pycache__/" in name
            or name.endswith(".pyc")
        ]
        if forbidden:
            raise RuntimeError(f"Forbidden files in Asset Library archive: {forbidden}")


def main(argv=None):
    args = parse_args(argv)
    version = validate_source()
    dist_dir = args.dist_dir.resolve()
    dist_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="metis-release-") as temporary:
        temporary = Path(temporary)
        wheel = build_wheel(args.python, temporary / "wheel")
        stage_root = temporary / "asset"
        stage_addon = stage_root / "addons" / "metis"
        _copy_addon(stage_addon)
        _write_runtime_payload(stage_addon, wheel, version)

        wheel_destination = dist_dir / wheel.name
        shutil.copy2(wheel, wheel_destination)
        archive_destination = dist_dir / f"metis-godot-{version}.zip"
        _write_deterministic_zip(stage_root, archive_destination)
        validate_archive(archive_destination, version)

    artifacts = [wheel_destination, archive_destination]
    checksum_path = dist_dir / "checksums.txt"
    checksum_path.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    print(f"Built Metis release {version}:")
    for path in [*artifacts, checksum_path]:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
