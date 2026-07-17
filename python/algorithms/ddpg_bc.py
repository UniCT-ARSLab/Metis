"""DDPG with behavior cloning training entrypoint."""

from algorithms.common import main as run_training


def main():
    run_training("ddpg_bc")


if __name__ == "__main__":
    main()
