from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import run_llm_benchmark, run_oracle_benchmark
from .datasets import download_and_prepare


def _stdout_utf8() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def download_main() -> None:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="Download and normalize GSM8K, MATH algebra, and PRM800K test data")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    print(json.dumps(download_and_prepare(args.data_dir, args.count, args.seed), ensure_ascii=False, indent=2))


def benchmark_main() -> None:
    _stdout_utf8()
    parser = argparse.ArgumentParser(description="Run the tri-dataset benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmark"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--mode", choices=["oracle", "llm"], default="oracle")
    parser.add_argument("--provider", choices=["openai", "wecodex", "siliconflow"], default="openai")
    parser.add_argument("--model")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--no-injected-failure", action="store_true")
    parser.add_argument("--limit-per-dataset", type=int)
    args = parser.parse_args()
    if args.mode == "oracle":
        report = run_oracle_benchmark(
            args.data_dir, args.output_dir, max_attempts=max(1, args.max_attempts),
            inject_first_failure=not args.no_injected_failure,
        )
    else:
        report = run_llm_benchmark(
            args.data_dir, args.output_dir, args.provider, args.model, args.max_attempts, args.limit_per_dataset
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


