"""Public command dispatcher for the Metis Python runtime."""

import importlib
import sys
from importlib import metadata


COMMAND_MODULES = {
    "train": "train",
    "run": "run",
    "record": "recorder",
    "export": "export",
    "doctor": "core.doctor",
}


def package_version() -> str:
    try:
        return metadata.version("metis-rl")
    except metadata.PackageNotFoundError:
        return "source"


def print_help() -> None:
    print(
        "Metis - Modular Environment for Training Intelligent Systems\n\n"
        "Usage:\n"
        "  metis <command> [options]\n\n"
        "Commands:\n"
        "  train    Train a policy against a Godot scenario\n"
        "  run      Run or evaluate a trained policy\n"
        "  record   Record manual demonstrations\n"
        "  export   Export a policy to deployment formats\n"
        "  doctor   Validate the local Metis runtime\n\n"
        "Run 'metis <command> --help' for command-specific options."
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print_help()
        return 0
    if argv[0] in {"-V", "--version", "version"}:
        print(f"metis-rl {package_version()}")
        return 0

    command = argv.pop(0)
    module_name = COMMAND_MODULES.get(command)
    if module_name is None:
        available = ", ".join(COMMAND_MODULES)
        raise SystemExit(
            f"Unknown Metis command {command!r}. Expected one of: {available}."
        )

    module = importlib.import_module(module_name)
    sys.argv = [f"metis {command}", *argv]
    result = module.main()
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
