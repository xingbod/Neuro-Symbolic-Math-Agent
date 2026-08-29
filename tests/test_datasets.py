import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import tempfile
import unittest

from neuro_symbolic_math_agent.benchmark import run_oracle_benchmark
from neuro_symbolic_math_agent.datasets import (
    BenchmarkRecord,
    _extract_boxed,
    create_fixed_benchmark,
    normalize_gsm8k,
    numeric_answer,
)


class DatasetAdapterTests(unittest.TestCase):
    def test_extracts_nested_boxed_answer(self):
        self.assertEqual(_extract_boxed(r"Thus $\boxed{\frac{3}{4}}$."), r"\frac{3}{4}")

    def test_numeric_answer_support(self):
        self.assertEqual(numeric_answer(r"\frac{3}{4}"), "3/4")
        self.assertEqual(numeric_answer(r"40,\!000"), "40000")
        self.assertIsNone(numeric_answer(r"\sqrt{2}"))

    def test_normalize_gsm8k(self):
        records = normalize_gsm8k([{"question": "1+1?", "answer": "Compute.\n#### 2"}])
        self.assertEqual(records[0].answer, "2")
        self.assertEqual(records[0].dataset, "gsm8k")

    def test_fixed_sample_is_reproducible(self):
        records = [BenchmarkRecord("x", str(i), "p", "s", str(i), ["step"], {}) for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            a = create_fixed_benchmark(records, Path(tmp) / "a.jsonl", 5, 42)
            b = create_fixed_benchmark(records, Path(tmp) / "b.jsonl", 5, 42)
        self.assertEqual([item.record_id for item in a], [item.record_id for item in b])

    def test_oracle_benchmark_runs_two_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            for name in ("gsm8k", "math", "prm800k"):
                records = [BenchmarkRecord(name, f"{name}-{i}", "1+1?", "2", "2", ["add"], {}) for i in range(50)]
                (data / f"{name}_50.jsonl").write_text(
                    "".join(json.dumps(record.__dict__ if hasattr(record, '__dict__') else {
                        "dataset": record.dataset, "record_id": record.record_id, "problem": record.problem,
                        "solution": record.solution, "answer": record.answer, "steps": record.steps,
                        "metadata": record.metadata,
                    }) + "\n" for record in records), encoding="utf-8"
                )
            report = run_oracle_benchmark(data, root / "results", max_attempts=2)
        self.assertEqual(report["overall"]["trials"], 300)
        self.assertEqual(report["overall"]["coverage"], 1.0)
        self.assertEqual(report["overall"]["mean_attempts"], 2.0)


if __name__ == "__main__":
    unittest.main()
