from __future__ import annotations

import argparse
import json

from .config import load_config
from .runner import BenchmarkRunner


def cmd_run(args):
    cfg = load_config(args.config)
    result = BenchmarkRunner(cfg).run()
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Side quest bi-encoder retrieval benchmark")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run model x dimension retrieval benchmark")
    p_run.add_argument("--config", required=True, help="Path to benchmark YAML config")
    p_run.set_defaults(func=cmd_run)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
