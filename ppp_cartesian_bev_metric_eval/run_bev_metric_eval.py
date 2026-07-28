from __future__ import annotations

import argparse

from bev_eval import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate marked PPP model exports in Cartesian BEV."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="bev_metric_eval_config.py",
        help="Path to the Python evaluation config.",
    )
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()
