from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from neuro_symbolic_math_agent.agent import MathAgent
from neuro_symbolic_math_agent.benchmark import TrialResult, summarize
from neuro_symbolic_math_agent.models import GenerationMode, Problem
from neuro_symbolic_math_agent.providers import LLMGenerator, OpenAICompatibleClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 3-item generation-only API smoke benchmark")
    parser.add_argument("--provider", choices=["wecodex", "siliconflow"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, default=Path("data/benchmark/formalized_smoke_3.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=1)
    args = parser.parse_args()

    client = OpenAICompatibleClient.from_env(args.provider, args.model)
    generator = LLMGenerator(client)
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.output.mkdir(parents=True, exist_ok=True)
    trials: list[TrialResult] = []
    started = time.perf_counter()

    with (
        (args.output / "trials.jsonl").open("w", encoding="utf-8", newline="\n") as trial_file,
        (args.output / "artifacts.jsonl").open("w", encoding="utf-8", newline="\n") as artifact_file,
    ):
        for case in cases:
            problem = Problem.from_dict(case["problem"])
            for mode in (GenerationMode.ISOMORPHIC, GenerationMode.SCAFFOLD):
                target = problem.steps[-1].step_id if mode == GenerationMode.SCAFFOLD else None
                trial_started = time.perf_counter()
                result = MathAgent(generator, max_attempts=args.max_attempts).run(problem, mode, target)
                verification = result.verification
                trial = TrialResult(
                    case["dataset"], case["record_id"], mode.value, result.delivered,
                    len(result.attempts), verification.cvi if verification else None,
                    verification.isomorphism_score if verification else None,
                    sorted({issue.code for attempt in result.attempts if attempt.verification for issue in attempt.verification.issues}),
                    result.abstention_reason,
                )
                trials.append(trial)
                trial_file.write(json.dumps({**asdict(trial), "elapsed_seconds": time.perf_counter() - trial_started}, ensure_ascii=False) + "\n")
                artifact_file.write(json.dumps({
                    "dataset": case["dataset"], "record_id": case["record_id"], "mode": mode.value,
                    "result": result.to_dict(),
                }, ensure_ascii=False) + "\n")
                trial_file.flush(); artifact_file.flush()

    report = summarize(trials)
    report["methodology"] = {
        "type": "three_item_generation_only_api_smoke",
        "provider": args.provider,
        "model": args.model,
        "formalized_cases": len(cases),
        "modes_per_case": 2,
        "max_attempts": args.max_attempts,
        "elapsed_seconds": time.perf_counter() - started,
        "note": "Uses cached formalization to skip the slow decomposition API stage.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
