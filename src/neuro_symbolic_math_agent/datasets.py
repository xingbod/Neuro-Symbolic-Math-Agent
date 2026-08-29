from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"
PRM800K_URL = "https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data/phase2_test.jsonl"


@dataclass(slots=True)
class BenchmarkRecord:
    dataset: str
    record_id: str
    problem: str
    solution: str
    answer: str
    steps: list[str]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkRecord":
        return cls(**data)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fetch_hf_split(dataset: str, config: str, split: str, output: Path, page_size: int = 100) -> list[dict[str, Any]]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    offset = 0
    total: int | None = None
    with requests.Session() as session:
        while total is None or offset < total:
            response = session.get(
                HF_ROWS_URL,
                params={"dataset": dataset, "config": config, "split": split, "offset": offset, "length": page_size},
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            total = int(payload["num_rows_total"])
            page = [entry["row"] for entry in payload["rows"]]
            rows.extend(page)
            if not page:
                break
            offset += len(page)
    _write_jsonl(output, rows)
    return rows


def download_file(url: str, output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with output.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
    return {"url": url, "path": str(output), "bytes": size, "sha256": digest.hexdigest()}


def _extract_boxed(solution: str) -> str:
    marker = "\\boxed{"
    starts = [match.start() for match in re.finditer(re.escape(marker), solution)]
    if not starts:
        return ""
    start = starts[-1] + len(marker)
    depth = 1
    for index in range(start, len(solution)):
        if solution[index] == "{":
            depth += 1
        elif solution[index] == "}":
            depth -= 1
            if depth == 0:
                return solution[start:index].strip()
    return ""


def _chosen_prm_steps(record: dict[str, Any]) -> list[str]:
    generated = record.get("question", {}).get("pre_generated_steps") or []
    if generated:
        return [str(step).strip() for step in generated if str(step).strip()]
    chosen: list[str] = []
    for step in record.get("label", {}).get("steps", []):
        index = step.get("chosen_completion")
        if index is None:
            text = step.get("human_completion")
        else:
            completions = step.get("completions", [])
            text = completions[index].get("text") if 0 <= index < len(completions) else None
        if text:
            chosen.append(str(text).strip())
    return chosen


def normalize_gsm8k(rows: list[dict[str, Any]]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for index, row in enumerate(rows):
        answer_blob = str(row["answer"])
        answer = answer_blob.rsplit("####", 1)[-1].strip()
        solution = answer_blob.rsplit("####", 1)[0].strip()
        records.append(BenchmarkRecord(
            "gsm8k", f"gsm8k-test-{index}", str(row["question"]), solution, answer,
            [line.strip() for line in solution.splitlines() if line.strip()], {"split": "test"},
        ))
    return records


def normalize_math(rows: list[dict[str, Any]]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for index, row in enumerate(rows):
        solution = str(row["solution"])
        records.append(BenchmarkRecord(
            "math", f"math-algebra-test-{index}", str(row["problem"]), solution, _extract_boxed(solution),
            [part.strip() for part in re.split(r"(?<=[.!?])\s+", solution) if part.strip()],
            {"split": "test", "config": "algebra", "level": row.get("level"), "type": row.get("type")},
        ))
    return records


def normalize_prm800k(rows: list[dict[str, Any]]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for index, row in enumerate(rows):
        question = row.get("question", {})
        records.append(BenchmarkRecord(
            "prm800k", f"prm800k-phase2-test-{index}", str(question.get("problem", "")),
            str(question.get("ground_truth_solution", "")), str(question.get("ground_truth_answer", "")),
            _chosen_prm_steps(row),
            {"split": "phase2_test", "generation": row.get("generation"), "finish_reason": row.get("label", {}).get("finish_reason")},
        ))
    return records


def numeric_answer(answer: str) -> str | None:
    value = answer.strip().replace(",", "").replace("\\!", "").replace("$", "")
    value = re.sub(r"^\\frac\{(-?\d+)\}\{(\d+)\}$", r"\1/\2", value)
    value = re.sub(r"^\\dfrac\{(-?\d+)\}\{(\d+)\}$", r"\1/\2", value)
    if re.fullmatch(r"-?\d+(?:/\d+)?", value):
        return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        return value
    return None


def create_fixed_benchmark(records: list[BenchmarkRecord], output: Path, count: int = 200, seed: int = 20260803) -> list[BenchmarkRecord]:
    eligible = [record for record in records if record.problem and numeric_answer(record.answer) is not None]
    if len(eligible) < count:
        raise ValueError(f"Only {len(eligible)} numeric-answer records available; need {count}.")
    random.Random(seed).shuffle(eligible)
    selected = eligible[:count]
    _write_jsonl(output, (asdict(record) for record in selected))
    return selected


def download_and_prepare(root: Path, count: int = 200, seed: int = 20260803) -> dict[str, Any]:
    raw = root / "raw"
    normalized = root / "normalized"
    benchmark = root / "benchmark"

    gsm_rows = fetch_hf_split("openai/gsm8k", "main", "test", raw / "gsm8k_test.jsonl")
    math_rows = fetch_hf_split("EleutherAI/hendrycks_math", "algebra", "test", raw / "math_algebra_test.jsonl")
    prm_meta = download_file(PRM800K_URL, raw / "prm800k_phase2_test.jsonl")
    prm_rows = _read_jsonl(raw / "prm800k_phase2_test.jsonl")

    groups = {
        "gsm8k": normalize_gsm8k(gsm_rows),
        "math": normalize_math(math_rows),
        "prm800k": normalize_prm800k(prm_rows),
    }
    manifest: dict[str, Any] = {
        "seed": seed,
        "sample_size_per_dataset": count,
        "sources": {
            "gsm8k": {"dataset": "openai/gsm8k", "config": "main", "split": "test"},
            "math": {"dataset": "EleutherAI/hendrycks_math", "config": "algebra", "split": "test"},
            "prm800k": prm_meta,
        },
        "datasets": {},
    }
    for name, records in groups.items():
        _write_jsonl(normalized / f"{name}.jsonl", (asdict(record) for record in records))
        selected = create_fixed_benchmark(records, benchmark / f"{name}_200.jsonl", count, seed)
        manifest["datasets"][name] = {
            "downloaded_records": len(records),
            "numeric_answer_records": sum(numeric_answer(record.answer) is not None for record in records),
            "selected_records": len(selected),
        }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_benchmark(path: Path) -> list[BenchmarkRecord]:
    return [BenchmarkRecord.from_dict(record) for record in _read_jsonl(path)]

