from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("experiments/tri_dataset_200_seed_20260803")
MODELS = {
    "gpt54mini": "GPT-5.4-Mini",
    "deepseek_v32": "DeepSeek-V3.2",
}
DATASETS = {
    "gsm8k": "GSM8K",
    "math": "MATH Algebra",
    "prm800k": "PRM800K",
}
MODES = {
    "isomorphic": "Isomorphic variant",
    "scaffold": "Step-focused scaffold",
}


ERROR_GROUPS = {
    "PARSE_ERROR": "Expression/schema parse failure",
    "ORIGINAL_PARSE_ERROR": "Source-structure parse failure",
    "DOMAIN_PARSE_ERROR": "Domain-constraint parse failure",
    "NO_SOLUTION_ERROR": "No finite complete solution",
    "SOLVER_ERROR": "Solver failure",
    "NON_REAL_SOLUTION_ERROR": "Non-real solution",
    "DOMAIN_VIOLATION_ERROR": "Semantic domain violation",
    "UGLY_SOLUTIONS_ERROR": "Low clean-value score",
    "NON_ISOMORPHIC_ERROR": "Structural mismatch",
    "TARGET_STEP_MISMATCH": "Scaffold target mismatch",
    "GENERATION_ERROR": "Generation or JSON failure",
    "PIPELINE_ERROR": "Pipeline or timeout failure",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100 * value:.2f}%"


def f3(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def trial_paths(model_slug: str) -> list[Path]:
    paths: list[Path] = []
    for dataset in DATASETS:
        paths.extend(sorted((ROOT / "trials" / model_slug / dataset).glob("*.json")))
    return paths


def load_trials(model_slug: str) -> list[dict[str, Any]]:
    return [load_json(path) for path in trial_paths(model_slug)]


def first_attempt(row: dict[str, Any]) -> dict[str, Any] | None:
    attempts = (row.get("result") or {}).get("attempts") or []
    return attempts[0] if attempts else None


def final_verification(row: dict[str, Any]) -> dict[str, Any] | None:
    return (row.get("result") or {}).get("verification")


def attempt_issue_codes(attempt: dict[str, Any] | None) -> list[str]:
    if not attempt:
        return []
    codes = []
    if attempt.get("generation_error"):
        codes.append("GENERATION_ERROR")
    verification = attempt.get("verification") or {}
    for issue in verification.get("issues") or []:
        code = issue.get("code")
        if code:
            codes.append(code)
    return codes


def row_issue_codes(row: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    if row.get("pipeline_error"):
        codes.append("PIPELINE_ERROR")
    for attempt in (row.get("result") or {}).get("attempts") or []:
        codes.extend(attempt_issue_codes(attempt))
    return codes


def last_failure_codes(row: dict[str, Any]) -> list[str]:
    if row.get("pipeline_error"):
        return ["PIPELINE_ERROR"]
    attempts = (row.get("result") or {}).get("attempts") or []
    if not attempts:
        return ["PIPELINE_ERROR"]
    return attempt_issue_codes(attempts[-1]) or ["UNKNOWN_FAILURE"]


def summarize_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trials = len(rows)
    first_candidates = 0
    first_valid = 0
    final_delivered = 0
    recovered = 0
    for row in rows:
        attempt = first_attempt(row)
        if attempt and attempt.get("candidate"):
            first_candidates += 1
            if (attempt.get("verification") or {}).get("accepted"):
                first_valid += 1
        if row.get("delivered"):
            final_delivered += 1
            if attempt and not (attempt.get("verification") or {}).get("accepted"):
                recovered += 1
    return {
        "trials": trials,
        "ungated_first_candidates": first_candidates,
        "ungated_first_candidate_valid": first_valid,
        "ungated_candidate_validity": first_valid / first_candidates if first_candidates else None,
        "ungated_valid_yield_per_trial": first_valid / trials if trials else None,
        "gate_only_coverage": first_valid / trials if trials else None,
        "gate_only_accepted_validity": 1.0 if first_valid else None,
        "gate_reflection_delivered": final_delivered,
        "gate_reflection_coverage": final_delivered / trials if trials else None,
        "gate_reflection_accepted_validity": 1.0 if final_delivered else None,
        "recovered_by_reflection": recovered,
        "absolute_coverage_gain": (final_delivered - first_valid) / trials if trials else None,
    }


def summarize_failures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempt_codes = Counter()
    abstention_primary = Counter()
    abstention_by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    abstention_by_mode: dict[str, Counter[str]] = defaultdict(Counter)
    representative: dict[str, dict[str, Any]] = {}

    for row in rows:
        for code in row_issue_codes(row):
            attempt_codes[ERROR_GROUPS.get(code, code)] += 1

        if row.get("delivered"):
            continue

        labels = [ERROR_GROUPS.get(code, code) for code in last_failure_codes(row)]
        primary = labels[0]
        abstention_primary[primary] += 1
        abstention_by_dataset[row["dataset"]][primary] += 1
        abstention_by_mode[row["mode"]][primary] += 1
        representative.setdefault(
            primary,
            {
                "dataset": row["dataset"],
                "record_id": row["record_id"],
                "mode": row["mode"],
                "codes": last_failure_codes(row),
                "pipeline_error": row.get("pipeline_error"),
            },
        )

    return {
        "attempt_issue_counts": dict(attempt_codes),
        "abstention_primary_counts": dict(abstention_primary),
        "abstention_primary_by_dataset": {k: dict(v) for k, v in abstention_by_dataset.items()},
        "abstention_primary_by_mode": {k: dict(v) for k, v in abstention_by_mode.items()},
        "representative_abstentions": representative,
    }


def summarize_isomorphism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not row.get("delivered"):
            continue
        verification = final_verification(row) or {}
        score = verification.get("isomorphism_score")
        if isinstance(score, (int, float)):
            values[row["mode"]].append(float(score))
    return {
        mode: {
            "n": len(scores),
            "mean_isomorphism_score": sum(scores) / len(scores) if scores else None,
            "reported_in_paper": sum(scores) / len(scores) if mode == "isomorphic" and scores else None,
        }
        for mode, scores in values.items()
    }


def summarize_formalization_errors() -> dict[str, Any]:
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    examples: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for path in sorted((ROOT / "formalization_errors" / dataset).glob("*.json")):
            item = load_json(path)
            error_type = item.get("error_type") or "Unknown"
            by_dataset[dataset][error_type] += 1
            if len(examples) < 5:
                examples.append(
                    {
                        "dataset": dataset,
                        "record_id": item.get("record_id"),
                        "error_type": error_type,
                        "error": item.get("error"),
                    }
                )
    return {
        "by_dataset": {k: dict(v) for k, v in by_dataset.items()},
        "examples": examples,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_tables(summary: dict[str, Any], stats: dict[str, Any], analysis: dict[str, Any]) -> str:
    overall_rows = []
    for slug, label in MODELS.items():
        overall = summary["models"][slug]["overall"]
        ci = stats["coverage_ci"][slug]["wilson_95"]
        iso = analysis["models"][slug]["isomorphism"].get("isomorphic", {})
        overall_rows.append(
            [
                label,
                f"{overall['delivered']}/{overall['trials']}",
                f"{pct(overall['coverage'])} [{pct(ci[0])}, {pct(ci[1])}]",
                pct(overall["accepted_output_validity"]),
                pct(overall["abstention_rate"]),
                f3(overall["mean_cvi"]),
                f3(iso.get("reported_in_paper")),
                f"{overall['mean_attempts']:.3f}",
            ]
        )

    dataset_rows = []
    for dataset, dataset_label in DATASETS.items():
        for slug, model_label in MODELS.items():
            item = summary["models"][slug]["by_dataset"][dataset]
            dataset_rows.append(
                [
                    dataset_label,
                    model_label,
                    f"{item['delivered']}/{item['trials']}",
                    pct(item["coverage"]),
                    f3(item["mean_cvi"]),
                ]
            )

    mode_rows = []
    for mode, mode_label in MODES.items():
        for slug, model_label in MODELS.items():
            item = summary["models"][slug]["by_mode"][mode]
            iso = analysis["models"][slug]["isomorphism"].get(mode, {})
            mode_rows.append(
                [
                    mode_label,
                    model_label,
                    f"{item['delivered']}/{item['trials']}",
                    pct(item["coverage"]),
                    f3(item["mean_cvi"]),
                    f3(iso.get("reported_in_paper")),
                ]
            )

    ablation_rows = []
    for slug, model_label in MODELS.items():
        item = analysis["models"][slug]["ablation"]
        ablation_rows.append(
            [
                model_label,
                pct(item["ungated_candidate_validity"]),
                pct(item["ungated_valid_yield_per_trial"]),
                pct(item["gate_only_coverage"]),
                pct(item["gate_reflection_coverage"]),
                pct(item["absolute_coverage_gain"]),
            ]
        )

    recovery_rows = []
    for slug, model_label in MODELS.items():
        item = stats["reflection_recovery"][slug]
        recovery_rows.append(
            [
                model_label,
                str(item["first_attempt_failures"]),
                str(item["recovered_on_retry"]),
                pct(item["recovery_rate"]),
            ]
        )

    failure_rows = []
    for slug, model_label in MODELS.items():
        counts = Counter(analysis["models"][slug]["failures"]["abstention_primary_counts"])
        for label, count in counts.most_common(8):
            failure_rows.append([model_label, label, str(count)])

    formal_rows = []
    formal = analysis["formalization_errors"]["by_dataset"]
    for dataset, dataset_label in DATASETS.items():
        total = sum(formal.get(dataset, {}).values())
        formal_rows.append([dataset_label, str(total), json.dumps(formal.get(dataset, {}), ensure_ascii=False)])

    sections = [
        "# Paper Analysis Tables",
        "",
        "## Overall Results",
        markdown_table(
            ["Model", "Delivered/Trials", "Coverage [95% CI]", "Controller-Gate Consistency", "Abstention", "Mean CVI", "Mean IS (isomorphic only)", "Mean Attempts"],
            overall_rows,
        ),
        "",
        "## Results by Dataset",
        markdown_table(["Dataset", "Model", "Delivered/Trials", "Coverage", "Mean CVI"], dataset_rows),
        "",
        "## Results by Mode",
        markdown_table(["Mode", "Model", "Delivered/Trials", "Coverage", "Mean CVI", "Mean IS (isomorphic only)"], mode_rows),
        "",
        "## Ablation from Existing Attempts",
        markdown_table(
            [
                "Model",
                "Ungated first-candidate validity",
                "Ungated valid yield/trial",
                "Gate only coverage",
                "Gate + reflection coverage",
                "Coverage gain",
            ],
            ablation_rows,
        ),
        "",
        "## Reflection Recovery",
        markdown_table(["Model", "First-attempt failures", "Recovered on retry", "Recovery rate"], recovery_rows),
        "",
        "## Primary Abstention Causes",
        markdown_table(["Model", "Cause", "Abstentions"], failure_rows),
        "",
        "## Formalization Failures",
        markdown_table(["Dataset", "Failures", "Error types"], formal_rows),
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    summary = load_json(ROOT / "summary.json")
    stats = load_json(ROOT / "statistical_analysis.json")

    analysis: dict[str, Any] = {"experiment_id": summary["experiment_id"], "models": {}}
    for slug in MODELS:
        rows = load_trials(slug)
        analysis["models"][slug] = {
            "ablation": summarize_ablation(rows),
            "failures": summarize_failures(rows),
            "isomorphism": summarize_isomorphism(rows),
        }
    analysis["formalization_errors"] = summarize_formalization_errors()

    (ROOT / "paper_analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "PAPER_TABLES.md").write_text(build_tables(summary, stats, analysis), encoding="utf-8")
    print(f"Wrote {ROOT / 'paper_analysis.json'}")
    print(f"Wrote {ROOT / 'PAPER_TABLES.md'}")


if __name__ == "__main__":
    main()
