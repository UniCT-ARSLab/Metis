#!/usr/bin/env python3
"""Open a packaged Metis add-on in an otherwise empty Godot project."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_TEMPLATE = ROOT / "packaging" / "smoke_project.godot"


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--godot-bin", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    archive = args.archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)

    with tempfile.TemporaryDirectory(prefix="metis-addon-smoke-") as temporary:
        project_dir = Path(temporary) / "project"
        home_dir = Path(temporary) / "home"
        data_dir = Path(temporary) / "data"
        project_dir.mkdir(parents=True)
        home_dir.mkdir()
        data_dir.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(project_dir)
        shutil.copy2(PROJECT_TEMPLATE, project_dir / "project.godot")

        environment = os.environ.copy()
        environment["HOME"] = str(home_dir)
        environment["XDG_DATA_HOME"] = str(data_dir)
        completed = subprocess.run(
            [
                args.godot_bin,
                "--headless",
                "--editor",
                "--path",
                str(project_dir),
                "--quit",
            ],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(completed.stdout, end="")
        script_failure_markers = (
            "SCRIPT ERROR:",
            "Parse Error:",
            "GDScript backtrace",
        )
        if completed.returncode != 0 or any(
            marker in completed.stdout for marker in script_failure_markers
        ):
            raise RuntimeError(
                "Godot rejected the packaged add-on "
                f"(exit code {completed.returncode})"
            )
    print(f"Packaged add-on smoke test passed: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
