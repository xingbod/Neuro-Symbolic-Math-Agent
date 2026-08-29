from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sympy as sp

from .agent import MathAgent
from .datasets import BenchmarkRecord, load_benchmark, numeric_answer
from .models import Candidate, DomainConstraint, GenerationMode, Problem, Step
from .providers import LLMGenerator, OpenAICompatibleClient, SequenceGenerator


@dataclass(slots=True)
class TrialResult:
    dataset: str
    record_id: str
    mode: str
    delivered: bool
    attempts: int
    cvi: float | None
    isomorphism_score: float | None
    error_codes: list[str]
    abstention_reason: str | None


class LLMDecomposer:
    def __init__(self, client: OpenAICompatibleClient) -> None:
        self.client = client

    def decompose(self, record: BenchmarkRecord) -> Problem:
        shape = {
            "problem_id": record.record_id,
            "text": record.problem,
            "variables": ["x"],
            "equations": ["x + 2 = 5"],
            "steps": [{
                "step_id": "s1", "name": "model", "description": "...", "equations": ["x+2=5"],
                "dependencies": [], "concepts": ["algebra"],
            }],
            "concepts": ["algebra"],
            "domain_constraints": [{"variable": "x", "expression": "x > 0", "require_all_solutions": True}],
        }
        payload = {
            "dataset": record.dataset,
            "problem": record.problem,
            "reference_solution": record.solution[:3000],
            "reference_answer": record.answer,
            "process_steps": [step[:500] for step in record.steps[:6]],
            "required_shape": shape,
        }
        data = self.client.complete_json(
            "Formalize the supplied solved mathematics item as strict JSON. Use SymPy syntax. equations must be a list "
            "of strings and include all numeric givens so every declared variable has a finite determined solution. "
            "Use exact integers or fractions, never decimal floats. domain_constraints are only inequalities or "
            "Ne(x, 0), never numeric assignments. Use at most 6 concise steps with descriptions under 120 characters. "
            "Build an acyclic step dependency graph. Do not change the problem.",
            json.dumps(payload, ensure_ascii=False),
        )
        problem = Problem.from_dict(data)
        if not problem.steps:
            problem.steps = [Step("s1", "solve", "Solve the formalized equations", list(problem.equations), [], list(problem.concepts))]
        return problem


def oracle_problem(record: BenchmarkRecord) -> Problem:
    answer = numeric_answer(record.answer)
    if answer is None:
        raise ValueError(f"Record {record.record_id} has no supported numeric answer")
    source_steps = record.steps or [record.solution or "Solve the problem"]
    steps = [
        Step(
            step_id=f"s{index + 1}",
            name=f"reasoning_step_{index + 1}",
            description=text[:1000],
            equations=[f"x={answer}"] if index == len(source_steps) - 1 else [],
            dependencies=[f"s{index}"] if index else [],
            concepts=[record.dataset],
        )
        for index, text in enumerate(source_steps[:20])
    ]
    return Problem(
        record.record_id,
        record.problem,
        ["x"],
        [f"x={answer}"],
        steps,
        [record.dataset, "numeric_answer"],
        [],
    )


def oracle_generator(record: BenchmarkRecord, target_step_id: str | None, inject_first_failure: bool) -> SequenceGenerator:
    answer = sp.Rational(numeric_answer(record.answer))
    valid = Candidate(
        question=f"Verified analogue of {record.record_id}: {record.problem}",
        variables=["x"],
        equations=[f"2*x={sp.sstr(2 * answer)}"],
        answer=f"x={sp.sstr(answer)}",
        target_step_id=target_step_id,
        metadata={"dataset": record.dataset, "source_id": record.record_id, "oracle_smoke_test": True},
    )
    if not inject_first_failure:
        return SequenceGenerator([valid])
    invalid = Candidate(
        question="Deliberately invalid reflection-loop probe",
        variables=["x"],
        equations=["x**2+1=0"],
        target_step_id=target_step_id,
        metadata={"deliberate_failure": True},
    )
    return SequenceGenerator([invalid, valid])


def _trial(agent_result: Any, record: BenchmarkRecord, mode: GenerationMode) -> TrialResult:
    codes = sorted({
        issue.code
        for attempt in agent_result.attempts
        if attempt.verification
        for issue in attempt.verification.issues
    })
    verification = agent_result.verification
    return TrialResult(
        record.dataset,
        record.record_id,
        mode.value,
        agent_result.delivered,
        len(agent_result.attempts),
        verification.cvi if verification else None,
        verification.isomorphism_score if verification else None,
        codes,
        agent_result.abstention_reason,
    )


def summarize(trials: list[TrialResult]) -> dict[str, Any]:
    def metrics(rows: list[TrialResult]) -> dict[str, Any]:
        delivered = [row for row in rows if row.delivered]
        return {
            "trials": len(rows),
            "delivered": len(delivered),
            "coverage": len(delivered) / len(rows) if rows else 0.0,
            "accepted_output_validity": 1.0 if delivered else None,
            "abstention_rate": 1 - len(delivered) / len(rows) if rows else 0.0,
            "mean_attempts": statistics.fmean(row.attempts for row in rows) if rows else 0.0,
            "mean_cvi": statistics.fmean(row.cvi for row in delivered if row.cvi is not None) if delivered else None,
            "mean_isomorphism_score": statistics.fmean(
                row.isomorphism_score for row in delivered if row.isomorphism_score is not None
            ) if delivered else None,
        }

    report: dict[str, Any] = {"overall": metrics(trials), "by_dataset": {}, "by_mode": {}}
    for dataset in sorted({trial.dataset for trial in trials}):
        report["by_dataset"][dataset] = metrics([trial for trial in trials if trial.dataset == dataset])
    for mode in sorted({trial.mode for trial in trials}):
        report["by_mode"][mode] = metrics([trial for trial in trials if trial.mode == mode])
    return report


def _resolve_benchmark_file(data_dir: Path, name: str) -> Path:
    for filename in (f"{name}_200.jsonl", f"{name}_50.jsonl", f"{name}.jsonl"):
        target = data_dir / filename
        if target.exists():
            return target
    raise FileNotFoundError(f"No benchmark file found for {name} in {data_dir}")


def run_oracle_benchmark(
    data_dir: Path,
    output_dir: Path,
    max_attempts: int = 2,
    limit_per_dataset: int | None = None,
    inject_first_failure: bool = True,
) -> dict[str, Any]:
    records = [
        record
        for name in ("gsm8k", "math", "prm800k")
        for record in load_benchmark(_resolve_benchmark_file(data_dir, name))[:limit_per_dataset]
    ]
    trials: list[TrialResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "oracle_trials.jsonl"
    with details_path.open("w", encoding="utf-8", newline="\n") as details:
        for record in records:
            problem = oracle_problem(record)
            for mode in (GenerationMode.ISOMORPHIC, GenerationMode.SCAFFOLD):
                target = problem.steps[0].step_id if mode == GenerationMode.SCAFFOLD else None
                generator = oracle_generator(record, target, inject_first_failure)
                result = MathAgent(generator, max_attempts=max_attempts).run(problem, mode, target)
                trial = _trial(result, record, mode)
                trials.append(trial)
                details.write(json.dumps(asdict(trial), ensure_ascii=False) + "\n")
    report = summarize(trials)
    report["methodology"] = {
        "type": "offline_oracle_verifier_smoke_benchmark",
        "warning": "This validates dataset adapters, verifier gates, reflection recovery, and aggregation; it does not measure LLM generation quality.",
        "inject_first_failure": inject_first_failure,
    }
    (output_dir / "oracle_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def run_llm_benchmark(
    data_dir: Path,
    output_dir: Path,
    provider: str,
    model: str | None,
    max_attempts: int = 5,
    limit_per_dataset: int | None = None,
) -> dict[str, Any]:
    client = OpenAICompatibleClient.from_env(provider, model)
    decomposer = LLMDecomposer(client)
    generator = LLMGenerator(client)
    records = [
        record
        for name in ("gsm8k", "math", "prm800k")
        for record in load_benchmark(_resolve_benchmark_file(data_dir, name))[:limit_per_dataset]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    trials: list[TrialResult] = []
    with (
        (output_dir / "llm_trials.jsonl").open("w", encoding="utf-8", newline="\n") as details,
        (output_dir / "llm_artifacts.jsonl").open("w", encoding="utf-8", newline="\n") as artifacts,
    ):
        for record in records:
            try:
                problem = decomposer.decompose(record)
            except Exception as exc:
                problem = None
                decomposition_error = str(exc)
            else:
                decomposition_error = None
            for mode in (GenerationMode.ISOMORPHIC, GenerationMode.SCAFFOLD):
                result = None
                try:
                    if problem is None:
                        raise RuntimeError(decomposition_error or "Problem decomposition failed")
                    if mode == GenerationMode.SCAFFOLD and not problem.steps:
                        raise ValueError("Decomposer returned no steps for scaffold mode")
                    target = problem.steps[-1].step_id if mode == GenerationMode.SCAFFOLD else None
                    result = MathAgent(generator, max_attempts=max_attempts).run(problem, mode, target)
                    trial = _trial(result, record, mode)
                except Exception as exc:
                    trial = TrialResult(
                        record.dataset, record.record_id, mode.value, False, 0, None, None,
                        ["PIPELINE_ERROR"], str(exc),
                    )
                trials.append(trial)
                details.write(json.dumps(asdict(trial), ensure_ascii=False) + "\n")
                artifacts.write(json.dumps({
                    "dataset": record.dataset,
                    "record_id": record.record_id,
                    "mode": mode.value,
                    "problem": asdict(problem) if problem else None,
                    "result": result.to_dict() if result else None,
                    "pipeline_error": trial.abstention_reason if result is None else None,
                }, ensure_ascii=False) + "\n")
                details.flush()
                artifacts.flush()
    report = summarize(trials)
    report["methodology"] = {"type": "llm_end_to_end", "provider": provider, "model": client.model}
    (output_dir / "llm_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report










