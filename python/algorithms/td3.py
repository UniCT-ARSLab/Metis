"""TD3 training entrypoint for continuous Godot action spaces."""

from algorithms.common import main as run_training


def main():
    run_training("td3")


if __name__ == "__main__":
    main()
