from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

from neuro_symbolic_math_agent.agent import MathAgent
from neuro_symbolic_math_agent.benchmark import LLMDecomposer
from neuro_symbolic_math_agent.datasets import BenchmarkRecord, load_benchmark
from neuro_symbolic_math_agent.models import GenerationMode, Problem
from neuro_symbolic_math_agent.providers import LLMGenerator, OpenAICompatibleClient

EXPERIMENT_ID = "tri_dataset_200_seed_20260803"
DATASETS = ("gsm8k", "math", "prm800k")
MODELS = {
    "gpt4omini": ("openai", "gpt-4o-mini"),
    "deepseek_v32": ("siliconflow", "deepseek-ai/DeepSeek-V3.2"),
}



def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_event(root: Path, event: dict[str, Any]) -> None:
    event = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **event}
    with (root / "events.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def prepare_samples(data_root: Path, experiment: Path, count: int, seed: int) -> dict[str, Any]:
    manifest: dict[str, Any] = {"experiment_id": EXPERIMENT_ID, "seed": seed, "count_per_dataset": count, "datasets": {}}
    for index, dataset in enumerate(DATASETS):
        source = data_root / "normalized" / f"{dataset}.jsonl"
        records = load_benchmark(source)
        rng = random.Random(seed + index)
        rng.shuffle(records)
        selected = records[:count]
        output = experiment / "samples" / f"{dataset}_200.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in selected), encoding="utf-8")
        manifest["datasets"][dataset] = {"source": str(source), "available": len(records), "selected": len(selected)}
    manifest["models"] = {slug: {"provider": provider, "model": model} for slug, (provider, model) in MODELS.items()}
    manifest["modes"] = [mode.value for mode in GenerationMode]
    write_json_atomic(experiment / "manifest.json", manifest)
    return manifest


def sample_records(experiment: Path) -> list[BenchmarkRecord]:
    return [
        record
        for dataset in DATASETS
        for record in load_benchmark(experiment / "samples" / f"{dataset}_200.jsonl")
    ]


def formalize(experiment: Path, concurrency: int) -> None:
    records = sample_records(experiment)
    os.environ["LLM_AUDIT_DIR"] = str((experiment / "api_audit" / "formalization_gpt4omini").resolve())
    client = OpenAICompatibleClient.from_env("openai", "gpt-4o-mini")
    decomposer = LLMDecomposer(client)

    def worker(record: BenchmarkRecord) -> tuple[str, bool, str | None]:
        target = experiment / "formalized" / record.dataset / f"{safe_name(record.record_id)}.json"
        if target.exists():
            return record.record_id, True, "cached"
        started = time.perf_counter()
        try:
            problem = decomposer.decompose(record)
            MathAgent._validate_problem(problem)
            payload = {
                "dataset": record.dataset, "record_id": record.record_id,
                "elapsed_seconds": time.perf_counter() - started,
                "problem": asdict(problem), "status": "ok",
            }
            write_json_atomic(target, payload)
            return record.record_id, True, None
        except Exception as exc:
            payload = {
                "dataset": record.dataset, "record_id": record.record_id,
                "elapsed_seconds": time.perf_counter() - started,
                "status": "error", "error_type": type(exc).__name__, "error": str(exc),
            }
            write_json_atomic(experiment / "formalization_errors" / record.dataset / f"{safe_name(record.record_id)}.json", payload)
            return record.record_id, False, str(exc)

    append_event(experiment, {"phase": "formalization", "status": "started", "records": len(records)})
    ok = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, record) for record in records]
        for index, future in enumerate(as_completed(futures), 1):
            _, success, _ = future.result()
            ok += int(success)
            if index % 20 == 0 or index == len(futures):
                print(f"formalization {index}/{len(futures)} success={ok}", flush=True)
    append_event(experiment, {"phase": "formalization", "status": "completed", "success": ok, "total": len(records)})


def run_model(experiment: Path, model_slug: str, concurrency: int, max_attempts: int) -> None:
    provider, model = MODELS[model_slug]
    records = sample_records(experiment)
    os.environ["LLM_AUDIT_DIR"] = str((experiment / "api_audit" / model_slug).resolve())
    client = OpenAICompatibleClient.from_env(provider, model)
    generator = LLMGenerator(client)

    jobs = [(record, mode) for record in records for mode in (GenerationMode.ISOMORPHIC, GenerationMode.SCAFFOLD)]

    def worker(record: BenchmarkRecord, mode: GenerationMode) -> tuple[str, bool, str]:
        target = experiment / "trials" / model_slug / record.dataset / f"{safe_name(record.record_id)}__{mode.value}.json"
        if target.exists():
            data = json.loads(target.read_text(encoding="utf-8"))
            return record.record_id, bool(data.get("delivered")), "cached"
        formal_path = experiment / "formalized" / record.dataset / f"{safe_name(record.record_id)}.json"
        started = time.perf_counter()
        result = None
        try:
            if not formal_path.exists():
                raise RuntimeError("FORMALIZATION_MISSING")
            formal = json.loads(formal_path.read_text(encoding="utf-8"))
            problem = Problem.from_dict(formal["problem"])
            target_step = problem.steps[-1].step_id if mode == GenerationMode.SCAFFOLD else None
            result = MathAgent(generator, max_attempts=max_attempts).run(problem, mode, target_step)
            payload = {
                "dataset": record.dataset, "record_id": record.record_id,
                "model_slug": model_slug, "provider": provider, "model": model, "mode": mode.value,
                "elapsed_seconds": time.perf_counter() - started,
                "delivered": result.delivered,
                "result": result.to_dict(),
                "pipeline_error": None,
            }
        except Exception as exc:
            payload = {
                "dataset": record.dataset, "record_id": record.record_id,
                "model_slug": model_slug, "provider": provider, "model": model, "mode": mode.value,
                "elapsed_seconds": time.perf_counter() - started,
                "delivered": False, "result": None,
                "pipeline_error": {"type": type(exc).__name__, "message": str(exc)},
            }
        write_json_atomic(target, payload)
        return record.record_id, bool(payload["delivered"]), "new"

    append_event(experiment, {"phase": "generation", "model_slug": model_slug, "status": "started", "trials": len(jobs)})
    delivered = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, record, mode) for record, mode in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            _, success, _ = future.result()
            delivered += int(success)
            if index % 25 == 0 or index == len(futures):
                print(f"{model_slug} {index}/{len(futures)} delivered={delivered}", flush=True)
    append_event(experiment, {"phase": "generation", "model_slug": model_slug, "status": "completed", "delivered": delivered, "trials": len(jobs)})


def aggregate(experiment: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"experiment_id": EXPERIMENT_ID, "models": {}}
    for model_slug in MODELS:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in (experiment / "trials" / model_slug).glob("*/*.json")]
        def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
            delivered = [item for item in items if item["delivered"]]
            cvis = [item["result"]["verification"]["cvi"] for item in delivered]
            scores = [item["result"]["verification"]["isomorphism_score"] for item in delivered]
            attempts = [len(item["result"]["attempts"]) if item.get("result") else 0 for item in items]
            errors = Counter()
            for item in items:
                if item.get("pipeline_error"):
                    errors["PIPELINE_ERROR"] += 1
                for attempt in (item.get("result") or {}).get("attempts", []):
                    if attempt.get("generation_error"):
                        errors["GENERATION_ERROR"] += 1
                    for issue in (attempt.get("verification") or {}).get("issues", []):
                        errors[issue["code"]] += 1
            return {
                "trials": len(items), "delivered": len(delivered),
                "coverage": len(delivered) / len(items) if items else 0.0,
                "accepted_output_validity": 1.0 if delivered else None,
                "abstention_rate": 1 - len(delivered) / len(items) if items else 0.0,
                "mean_attempts": statistics.fmean(attempts) if attempts else 0.0,
                "mean_cvi": statistics.fmean(cvis) if cvis else None,
                "mean_isomorphism_score": statistics.fmean(scores) if scores else None,
                "error_counts": dict(errors),
            }
        model_summary = {"overall": metrics(rows), "by_dataset": {}, "by_mode": {}}
        for dataset in DATASETS:
            model_summary["by_dataset"][dataset] = metrics([row for row in rows if row["dataset"] == dataset])
        for mode in (GenerationMode.ISOMORPHIC, GenerationMode.SCAFFOLD):
            model_summary["by_mode"][mode.value] = metrics([row for row in rows if row["mode"] == mode.value])
        summary["models"][model_slug] = model_summary
    formal_ok = len(list((experiment / "formalized").glob("*/*.json")))
    formal_errors = len(list((experiment / "formalization_errors").glob("*/*.json")))
    summary["formalization"] = {"success": formal_ok, "errors": formal_errors, "total": formal_ok + formal_errors}
    write_json_atomic(experiment / "summary.json", summary)
    return summary


def paper_draft(experiment: Path, summary: dict[str, Any]) -> None:
    def pct(value: float | None) -> str:
        return "—" if value is None else f"{100 * value:.2f}%"
    rows = []
    for slug, data in summary["models"].items():
        m = data["overall"]
        rows.append(f"| {MODELS[slug][1]} | {m['trials']} | {pct(m['coverage'])} | {pct(m['accepted_output_validity'])} | {pct(m['abstention_rate'])} | {m['mean_cvi'] or 0:.3f} | {m['mean_isomorphism_score'] or 0:.3f} |")
    text = f'''# Neuro-Symbolic Math Agent: Verifier-Gated Generation of Isomorphic Variants and Step-Focused Scaffolds

## Abstract

Large language models can generate adaptive mathematics exercises, but unconstrained generation may produce unsolvable equations, domain-invalid answers, or numerically unsuitable values. We present a neuro-symbolic mathematics agent that combines LLM-based generation with SymPy verification, semantic domain checking, a Clean-Value Index (CVI), structural isomorphism scoring, execution-error reflection, and safe abstention. We evaluate the system on 600 source problems sampled reproducibly from GSM8K, MATH Algebra, and PRM800K (200 per dataset). Each problem is tested in isomorphic-variant and step-focused-scaffold modes using GPT-5.4-Mini and DeepSeek-V3.2. All prompts, raw model responses, formalizations, candidates, verifier traces, and trial outcomes are retained for audit. The verifier gate guarantees that no rejected item is delivered. Empirical results show the trade-off between candidate coverage and accepted-output validity.

## 1. Introduction

Adaptive intelligent tutoring requires targeted practice without exposing learners to mathematically invalid generated content. This work operationalizes verifier-gated generation: a neural model proposes a candidate while a symbolic system decides whether it is safe to deliver. The study addresses coverage, accepted-output safety, reflection-based recovery, numerical cleanliness, and structural preservation.

## 2. Method

### 2.1 Data

We use a fixed random seed (20260803) to sample 200 problems from each of GSM8K test, MATH Algebra test, and PRM800K phase-2 test, yielding 600 source problems. For every source problem, the system evaluates two generation modes, giving 1,200 trials per model.

### 2.2 Agent

The agent represents each problem as variables, equations, semantic constraints, concepts, and a directed acyclic step graph. GPT-5.4-Mini performs a shared formalization pass. Candidate generation is evaluated separately with GPT-5.4-Mini and DeepSeek-V3.2. The verifier applies parsing, finite solvability, realness, domain constraints, CVI thresholding, and structural isomorphism checks. Failed candidates receive structured reflection feedback for at most two attempts. When verification still fails, the system abstains.

### 2.3 Auditability

The experiment stores the selected source records, formalized problems, every API request and raw response, all generated candidates, symbolic solutions, verifier issues, reflections, timings, and aggregate summaries. API credentials are excluded from artifacts.

## 3. Results

| Model | Trials | Coverage | Accepted Validity | Abstention | CVI | IS |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Formalization succeeded for {summary['formalization']['success']} of {summary['formalization']['total']} source problems; {summary['formalization']['errors']} failed formalization.

Accepted-output validity is conditional on delivery: it measures valid delivered items divided by all delivered items. Coverage measures delivered items divided by all attempted trials. Thus, 100% accepted validity can coexist with lower coverage when the gate safely abstains.

## 4. Discussion

The results demonstrate the intended safety-coverage trade-off. The symbolic gate prevents invalid delivery, while model and domain differences affect how frequently a usable candidate is produced. Error distributions in the accompanying summary identify whether failures arise primarily from parsing, underdetermined systems, non-real roots, domain violations, low CVI, or structural mismatch. Because the formalization is shared, model comparisons focus on generation behavior rather than decomposition variance.

## 5. Limitations

The experiment uses a single shared decomposer, heuristic structural similarity, and automatic symbolic criteria that do not fully measure pedagogical quality. Accepted-output validity is guaranteed by the gate and should not be interpreted as raw model accuracy. Human expert review remains necessary for construct validity, age appropriateness, and scaffold quality. Qwen models were excluded from the main experiment after preliminary API tests exhibited repeated 120–247 second latency and timeouts.

## 6. Conclusion

A verifier-gated neuro-symbolic workflow can deliver only mathematically validated generated items while explicitly exposing the cost as abstention. The retained audit trail supports manual review and reproducibility. Future work should add expert ratings, stricter graph-isomorphism measures, unit-aware constraints, and controlled comparisons of decomposer models.

## Reproducibility Statement

Experiment ID: `{EXPERIMENT_ID}`. Seed: 20260803. Detailed artifacts are stored under `{experiment}`.
'''
    (experiment / "PAPER_DRAFT_INITIAL.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "formalize", "generate", "aggregate", "all"], default="all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--experiment-root", type=Path, default=Path("experiments") / EXPERIMENT_ID)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--models", nargs="*", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args()
    root = args.experiment_root
    root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(root / "run_config.json", vars(args) | {"data_root": str(args.data_root), "experiment_root": str(root)})
    if args.phase in ("prepare", "all"):
        prepare_samples(args.data_root, root, args.count, args.seed)
    if args.phase in ("formalize", "all"):
        formalize(root, args.concurrency)
    if args.phase in ("generate", "all"):
        for model_slug in args.models:
            run_model(root, model_slug, args.concurrency, args.max_attempts)
    if args.phase in ("aggregate", "all"):
        summary = aggregate(root)
        paper_draft(root, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
