from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sympy as sp

from neuro_symbolic_math_agent.models import Candidate

ROOT = Path("experiments/tri_dataset_200_seed_20260803")
MODELS = ("gpt54mini", "deepseek_v32")
DATASETS = ("gsm8k", "math", "prm800k")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_expr(text: str, symbols: dict[str, sp.Symbol] | None = None) -> sp.Expr | None:
    text = text.strip().replace("^", "**")
    text = re.sub(r"\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = text.replace("\\sqrt", "sqrt")
    try:
        return sp.simplify(sp.sympify(text, locals=symbols or {}, rational=True))
    except Exception:
        return None


def answer_status(candidate: dict[str, Any], solutions: list[dict[str, str]]) -> tuple[str, str]:
    answer = str(candidate.get("answer") or "").strip()
    variables = [str(v) for v in candidate.get("variables") or []]
    if not answer:
        return "missing", "empty answer"
    syms = {name: sp.Symbol(name, real=True) for name in variables}
    solved = {name: [parse_expr(sol[name], syms) for sol in solutions if name in sol] for name in variables}
    assignments_found = False
    matched = False
    contradicted: list[str] = []
    for name in variables:
        pattern = rf"(?:^|[,;\n(:])\s*{re.escape(name)}\s*=\s*([^,;\n]+)"
        for match in re.finditer(pattern, answer):
            assignments_found = True
            segment = re.split(r"\bor\b|\\text\{|\band\b", match.group(1), maxsplit=1, flags=re.I)[0]
            parts = [p.strip() for p in segment.split("=") if p.strip()]
            parsed = [parse_expr(part, syms) for part in parts]
            parsed = [value for value in parsed if value is not None and not value.free_symbols]
            expected = [value for value in solved.get(name, []) if value is not None]
            if any(sp.simplify(value - target) == 0 for value in parsed for target in expected):
                matched = True
            elif parsed and expected:
                contradicted.append(f"{name}={parts[-1]}")
    if matched:
        return "matched", "declared assignment agrees with a solved value"
    if assignments_found and contradicted:
        return "contradicted", "; ".join(contradicted)

    if len(variables) > 1 and answer.startswith("(") and answer.endswith(")"):
        try:
            values = ast.literal_eval(answer)
            parts = [parse_expr(str(v), syms) for v in values]
        except Exception:
            parts = []
        for solution in solutions:
            expected = [parse_expr(solution.get(name, ""), syms) for name in variables]
            if len(parts) == len(expected) and all(
                parts[i] is not None and expected[i] is not None and sp.simplify(parts[i] - expected[i]) == 0
                for i in range(len(parts))
            ):
                return "matched", "ordered tuple agrees with a solution"

    if len(variables) == 1:
        value = parse_expr(answer, syms)
        expected = [v for v in solved[variables[0]] if v is not None]
        if value is not None and not value.free_symbols:
            if any(sp.simplify(value - target) == 0 for target in expected):
                return "matched", "bare value agrees with a solved value"
            return "contradicted", f"bare value {answer!r} disagrees with solved values"
    return "unverified", "answer format could not be checked mechanically"


def response_content(audit: dict[str, Any]) -> dict[str, Any] | None:
    if audit.get("status") != "ok":
        return None
    try:
        text = audit["response"]["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
        return Candidate.from_dict(value).to_dict() if isinstance(value, dict) else None
    except Exception:
        return None


def main() -> None:
    formalized = list((ROOT / "formalized").glob("*/*.json"))
    formalization_errors = list((ROOT / "formalization_errors").glob("*/*.json"))

    api_hashes: dict[str, Counter[str]] = {}
    api_stats: dict[str, Any] = {}
    for model in MODELS:
        hashes: Counter[str] = Counter()
        statuses: Counter[str] = Counter()
        response_ids: list[str] = []
        created_values: list[int] = []
        for path in (ROOT / "api_audit" / model).glob("*.json"):
            row = load(path)
            statuses[row.get("status", "missing")] += 1
            content = response_content(row)
            if content is not None:
                hashes[hashlib.sha256(canonical(content).encode()).hexdigest()] += 1
            response = row.get("response") or {}
            if response.get("id"):
                response_ids.append(str(response["id"]))
            if isinstance(response.get("created"), int):
                created_values.append(response["created"])
        api_hashes[model] = hashes
        api_stats[model] = {
            "files": sum(statuses.values()), "statuses": dict(statuses),
            "decodable_json_responses": sum(hashes.values()), "unique_json_responses": len(hashes),
            "response_ids": len(response_ids), "unique_response_ids": len(set(response_ids)),
            "created_epoch_min": min(created_values) if created_values else None,
            "created_epoch_max": max(created_values) if created_values else None,
        }

    model_report: dict[str, Any] = {}
    paired: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
    examples: list[dict[str, Any]] = []

    for model in MODELS:
        paths = sorted((ROOT / "trials" / model).glob("*/*.json"))
        rows = [load(path) for path in paths]
        delivered_rows = [row for row in rows if row.get("delivered")]
        by_dataset = Counter(row["dataset"] for row in delivered_rows)
        by_mode = Counter(row["mode"] for row in delivered_rows)
        attempts = candidate_attempts = audit_matched = 0
        delivery_invariant_failures = 0
        answer_counts: Counter[str] = Counter()
        cvis: list[float] = []
        iso_values: list[float] = []
        first_candidates = first_valid = recovered = 0

        for row in rows:
            paired[(row["dataset"], row["record_id"], row["mode"])][model] = bool(row.get("delivered"))
            result = row.get("result") or {}
            row_attempts = result.get("attempts") or []
            attempts += len(row_attempts)
            if row_attempts and row_attempts[0].get("candidate"):
                first_candidates += 1
                if (row_attempts[0].get("verification") or {}).get("accepted"):
                    first_valid += 1
            if row.get("delivered") and row_attempts and not (row_attempts[0].get("verification") or {}).get("accepted"):
                recovered += 1
            for attempt in row_attempts:
                candidate_data = attempt.get("candidate")
                if not candidate_data:
                    continue
                candidate_attempts += 1
                digest = hashlib.sha256(canonical(Candidate.from_dict(candidate_data).to_dict()).encode()).hexdigest()
                audit_matched += int(bool(api_hashes[model][digest]))
            if row.get("delivered"):
                verification = result.get("verification") or {}
                final_attempt = row_attempts[-1] if row_attempts else {}
                if not verification.get("accepted") or not (final_attempt.get("verification") or {}).get("accepted"):
                    delivery_invariant_failures += 1
                if result.get("candidate") != final_attempt.get("candidate"):
                    delivery_invariant_failures += 1
                cvis.append(float(verification.get("cvi", 0)))
                if row["mode"] == "isomorphic":
                    iso_values.append(float(verification.get("isomorphism_score", 0)))
                status, reason = answer_status(result["candidate"], verification.get("solutions") or [])
                answer_counts[status] += 1
                if status == "contradicted" and len(examples) < 30:
                    examples.append({
                        "model": model, "dataset": row["dataset"], "record_id": row["record_id"],
                        "mode": row["mode"], "answer": result["candidate"].get("answer"),
                        "solutions": verification.get("solutions"), "reason": reason,
                        "trial_file": str(ROOT / "trials" / model / row["dataset"] / f'{row["record_id"]}__{row["mode"]}.json'),
                    })

        model_report[model] = {
            "trial_files": len(rows),
            "unique_trial_keys": len({(r["dataset"], r["record_id"], r["mode"]) for r in rows}),
            "delivered": len(delivered_rows), "coverage": len(delivered_rows) / len(rows),
            "delivered_by_dataset": dict(by_dataset), "delivered_by_mode": dict(by_mode),
            "logged_attempts": attempts, "candidate_attempts": candidate_attempts,
            "candidate_attempts_matched_to_raw_api_json": audit_matched,
            "delivery_invariant_failures": delivery_invariant_failures,
            "first_candidates": first_candidates, "first_valid": first_valid,
            "recovered_by_retry": recovered,
            "answer_check": dict(answer_counts),
            "logged_mean_cvi": sum(cvis) / len(cvis) if cvis else None,
            "logged_mean_isomorphism_isomorphic_only": sum(iso_values) / len(iso_values) if iso_values else None,
        }

    discordant = Counter()
    complete_pairs = 0
    for outcomes in paired.values():
        if set(outcomes) == set(MODELS):
            complete_pairs += 1
            if outcomes[MODELS[0]] and not outcomes[MODELS[1]]:
                discordant["gpt_only"] += 1
            elif outcomes[MODELS[1]] and not outcomes[MODELS[0]]:
                discordant["deepseek_only"] += 1
    n = discordant["gpt_only"] + discordant["deepseek_only"]
    smaller = min(discordant["gpt_only"], discordant["deepseek_only"])
    exact_p = min(1.0, 2 * sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)) if n else 1.0

    report = {
        "experiment_root": str(ROOT),
        "formalization_files": len(formalized), "formalization_error_files": len(formalization_errors),
        "models": model_report,
        "paired": {"complete_pairs": complete_pairs, **dict(discordant), "exact_mcnemar_p": exact_p},
        "api_audit": api_stats,
        "confirmed_answer_mismatch_examples": examples,
        "method_limits": [
            "Raw API logs are local JSON without provider signatures or cryptographic attestation.",
            "The verifier never compares candidate.answer with its symbolic solutions.",
            "Candidate question text and equation semantics are not machine-validated.",
            "The answer checker is conservative; unparseable cases are not counted as errors.",
        ],
    }
    out = ROOT / "independent_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

