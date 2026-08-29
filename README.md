# Neuro-Symbolic Math Agent

> A Verifier-Gated Framework for Mathematics Item Generation, SymPy Verification, and Reflective Remediation.

---

## Description

The **Neuro-Symbolic Math Agent** is a pedagogical framework designed for generating, verifying, and refining mathematics items for educational assessment. The system integrates Large Language Models (LLMs) for natural language item generation with the **SymPy symbolic computation engine** for rigorous mathematical verification.

Operating under a **Verifier-Gated Controller** architecture, the system formalizes source items into step dependency graphs (DAGs), generates candidate variants (either isomorphic or step-focused scaffolds), and subjects all candidates to formal symbolic checks. If verification fails, a **Reflection Engine** converts diagnostic error codes into prompt-level corrective constraints for iterative retry attempts (up to $k$ max retries). If all attempts fail, the controller enforces a **fail-safe abstention** policy (`delivered = false`), preventing unverified or mathematically invalid items from being delivered to students.

---

## Dataset Information

The repository includes structured datasets for evaluating math item generation across three benchmark domains:

- **Paper Benchmark Set (`data/benchmark/`)**:
  - `gsm8k_200.jsonl` (200 items): Grade-school arithmetic word problems, rate/time, and applied multi-step math.
  - `math_200.jsonl` (200 items): High-school algebra, quadratic equations, polynomial systems, and domain-constrained expressions.
  - `prm800k_200.jsonl` (200 items): Multi-step process-supervised reasoning problems with step-dependency annotations.

- **Full Normalized Splits (`data/normalized/`)**:
  - `gsm8k.jsonl` (1,319 items): Full normalized GSM8K test split.
  - `math.jsonl` (1,187 items): Full normalized MATH algebra test split.
  - `prm800k.jsonl` (4,500 items): Full normalized PRM800K Phase-2 test split.

Each record includes `dataset`, `record_id`, `problem` text, reference `solution`, ground-truth `answer`, process `steps`, and domain `metadata`.

---

## Code Information

```text
release/
├── pyproject.toml                     # Python package specification (pip install -e .)
├── config.json                        # API Base URL & Key configuration file (git-ignored)
├── .env.example                       # Environment variable configuration template
├── README.md                          # English documentation (this file)
├── README.zh-CN.md                    # Chinese documentation
├── src/                               # Core engine package
│   └── neuro_symbolic_math_agent/
│       ├── __init__.py
│       ├── agent.py                   # MathAgent controller & verification-gated loop
│       ├── verifier.py                # SymPy solver, real-root, domain, CVI & SFS checker
│       ├── reflection.py              # Reflection Engine mapping error codes to prompts
│       ├── models.py                  # Problem, Step, Candidate, Verification datatypes
│       ├── datasets.py                # Dataset downloaders, normalizers, and samplers
│       ├── providers.py               # OpenAI & SiliconFlow stdlib API clients
│       ├── benchmark.py               # Oracle & LLM benchmark evaluation engine
│       ├── cli.py                     # Interactive single-item CLI entrypoint
│       └── benchmark_cli.py           # Benchmark execution CLI entrypoint
├── web/                               # Interactive Web Application
│   ├── server.py                      # FastAPI REST server (/api/config, /api/datasets, /api/run)
│   ├── templates/
│   │   └── index.html                 # Responsive dual-panel Web UI
│   └── static/
│       ├── app.css                    # Modern UI styling & metric gauges
│       └── app.js                     # Frontend AJAX & visual execution trace renderer
├── examples/                          # Demonstration code
│   ├── demo_run.py                    # Dual-mode (Offline + Online) Python demo script
│   └── quadratic_problem.json         # Structured JSON input example
├── tests/                             # Unit tests
│   ├── test_agent.py                  # Verifier, reflection loop & abstention unit tests
│   └── test_datasets.py               # Dataset adapter & benchmark runner unit tests
├── scripts/                           # Experiment & paper analysis tools
│   ├── run_large_experiment.py        # 200-item 3-dataset benchmark execution script
│   ├── run_small_api_benchmark.py     # Quick API benchmark runner
│   ├── analyze_experiment_for_paper.py# Statistical analysis (Pass Rate, CVI, SFS, McNemar CI)
│   └── audit_paper_results.py         # Paper results reproducibility auditor
└── data/                              # Benchmark & normalized datasets
    └── benchmark/                     # 200-item benchmark JSONL files
```

---

## Usage Instructions

### 1. Installation

Python 3.10+ is required. Install the package in editable mode:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

# Install package
pip install -e .
```

### 2. API Key Configuration (`config.json`)

Configure API credentials directly in `config.json` at the root directory:

```json
{
  "openai": {
    "api_key": "your-openai-api-key-here",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
  },
  "siliconflow": {
    "api_key": "your-siliconflow-api-key-here",
    "base_url": "https://api.siliconflow.cn/v1",
    "model": "deepseek-ai/DeepSeek-V3"
  }
}
```

### 3. Running the Interactive Web UI

Launch the FastAPI web server to browse the 200-item dataset and visualize execution traces:

```bash
python -m uvicorn web.server:app --host 0.0.0.0 --port 8000
```

Open **`http://localhost:8000`** in your browser to:
- Browse and search all 200 items in GSM8K, MATH, and PRM800K.
- Configure API Keys/URLs or toggle Offline Mock Mode.
- Visualize Problem DAGs, Attempt Loops, Verifier Reflection Prompts, CVI/SFS scores, and final delivery status.

### 4. Running Python Demo Code

Run the offline/online dual-mode demo script:

```bash
python examples/demo_run.py
```

### 5. Running Unit Tests & Benchmarks

```bash
# Run unit tests
python -m unittest discover -s tests

# Run 300-trial offline oracle benchmark
math-agent-benchmark --mode oracle --max-attempts 2

# Resample benchmark dataset (e.g., 200 items)
math-agent-download --count 200 --seed 20260803
```

---

## Requirements

The project relies on standard Python libraries and lightweight dependencies:

- **Python**: $\ge 3.10$
- **SymPy**: $\ge 1.12$ (Symbolic mathematical parsing, solving, and domain checking)
- **FastAPI**: $\ge 0.100.0$ (Web REST API server)
- **Uvicorn**: $\ge 0.22.0$ (ASGI server)
- **Pydantic**: $\ge 2.0.0$ (Data validation)
- **Jinja2**: $\ge 3.0.0$ (Web template rendering)
- **Requests**: $\ge 2.28.0$ (HTTP client for dataset downloading)

---

## Methodology

The framework follows a four-stage neuro-symbolic pipeline:

1. **Formalization (Decomposer)**: Natural language math items are parsed into structured `Problem` instances with declared variables, SymPy-compatible equations, step dependency DAGs (`Step`), and domain constraints (`DomainConstraint`).
2. **Item Generation**:
   - **`isomorphic`**: Generates variants maintaining variable count, equation degree, operation complexity, and step logic while altering surface context and numerical coefficients.
   - **`scaffold`**: Generates reduced-dimension transition items isolating a specific stuck step (`target_step_id`).
3. **Symbolic Verification (`SymbolicVerifier`)**:
   - **Solvability & Real Roots**: Computes exact SymPy solutions and verifies finite real solvability.
   - **Domain Constraints**: Validates solutions against inequalities (e.g., $x > 0$, $x \neq 0$).
   - **Clean-Value Index (CVI)**: Scores pedagogical value suitability (favoring small integers and simple fractions; penalizing decimal floats and ugly radicals).
   - **Symbolic Feature-Similarity (SFS)**: Evaluates structural isomorphism ($\ge 0.60$ threshold).
4. **Reflective Remediation & Abstention**: Upon verification failure, the `ReflectionEngine` maps error taxonomy codes (`NO_SOLUTION_ERROR`, `DOMAIN_VIOLATION_ERROR`, `UGLY_SOLUTIONS_ERROR`, `NON_ISOMORPHIC_ERROR`) into natural-language prompt constraints for up to $k$ retries. If all attempts fail, `delivered` is set to `false`.

---

## Citations

If you use this codebase or benchmark dataset in your research, please cite:

```bibtex
@article{neuro_symbolic_math_agent_2026,
  title={Verifier-Gated Neuro-Symbolic Agent for Pedagogical Mathematics Item Generation and Reflective Remediation},
  author={Antigravity Team},
  journal={PeerJ Computer Science Draft},
  year={2026}
}
```

Additionally, please reference the original datasets:

- **GSM8K**: Cobbe et al., 2021. *Training Verifiers to Solve Math Word Problems*. [arXiv:2110.14168](https://arxiv.org/abs/2110.14168).
- **MATH**: Hendrycks et al., 2021. *Measuring Mathematical Problem Solving With the MATH Dataset*. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874).
- **PRM800K**: Lightman et al., 2023. *Let's Verify Step by Step*. [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).

---

## License & Contribution Guidelines

### License
This project is licensed under the **MIT License**. The underlying datasets (GSM8K, MATH, PRM800K) are subject to their respective original open-source licenses.

### Contribution Guidelines
1. **Pull Requests**: We welcome contributions! Please ensure all code changes include corresponding unit tests in `tests/`.
2. **Verification Rules**: When adding new SymPy domain checks or CVI metrics, ensure backwards compatibility with existing `Problem` dataclasses.
3. **Security**: **Do not commit API Keys or `.env` / `config.json` files containing live credentials.** Use `config.json.example` as reference.
