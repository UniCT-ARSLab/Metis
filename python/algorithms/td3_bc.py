"""TD3 with behavior cloning training entrypoint."""

from algorithms.common import main as run_training


def main():
    run_training("td3_bc")


if __name__ == "__main__":
    main()
