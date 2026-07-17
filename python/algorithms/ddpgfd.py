"""DDPG from Demonstrations training entrypoint."""

from algorithms.common import main as run_training


def main():
    run_training("ddpgfd")


if __name__ == "__main__":
    main()
