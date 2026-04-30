from __future__ import annotations

import argparse

from .config import load_config
from .experiment import ExperimentRunner
from .gpu_preflight import validate_cuda_runtime_or_raise
from .training.sbert_trainer import SbertMatryoshkaFinetuner
from .logging_utils import configure_logging


def cmd_train(args):
    config = load_config(args.config)
    if str(config.execution.device).strip().lower() == "cuda":
        validate_cuda_runtime_or_raise(context="cli.train")
    logger = configure_logging(verbose=True)
    finetuner = SbertMatryoshkaFinetuner(config, logger)
    finetuner.run()


def cmd_run(args):
    config = load_config(args.config)
    wants_cuda = (
        str(config.execution.device).strip().lower() == "cuda"
        or str(config.execution.retrieval_device).strip().lower() == "cuda"
    )
    if wants_cuda:
        validate_cuda_runtime_or_raise(context="cli.run")
    runner = ExperimentRunner(config)
    runner.run()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matryoshka retrieval experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Fine-tune a SentenceTransformer model")
    p_train.add_argument("--config", required=True, help="Path to a YAML config file")
    p_train.set_defaults(func=cmd_train)

    p_run = sub.add_parser("run", help="Run the full retrieval experiment")
    p_run.add_argument("--config", required=True, help="Path to a YAML config file")
    p_run.set_defaults(func=cmd_run)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
