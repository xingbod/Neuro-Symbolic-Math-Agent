from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import MathAgent
from .models import Candidate, DomainConstraint, GenerationMode, Problem, Step
from .providers import LLMGenerator, OpenAICompatibleClient, SequenceGenerator


def demo_problem() -> Problem:
    return Problem(
        problem_id="quadratic-01",
        text="A rectangle has side lengths x-2 and x-3 and area 0. Find the boundary values of x.",
        variables=["x"],
        equations=["x**2 - 5*x + 6 = 0"],
        steps=[
            Step("s1", "model", "Expand the product", ["(x-2)*(x-3)=0"], concepts=["modeling"]),
            Step("s2", "solve", "Solve the quadratic", ["x**2-5*x+6=0"], ["s1"], ["quadratics"]),
        ],
        concepts=["quadratics", "factoring"],
        domain_constraints=[DomainConstraint("x", "x > 0")],
    )


def demo_generator() -> SequenceGenerator:
    return SequenceGenerator([
        Candidate(
            "A square parameter satisfies x²+x+1=0. Find x.", ["x"], ["x**2+x+1=0"],
            domain_constraints=[DomainConstraint("x", "x > 0")],
        ),
        Candidate(
            "A product (x-3)(x-4) is zero. Find the two values of x.", ["x"], ["x**2-7*x+12=0"],
            answer="x = 3 or x = 4", domain_constraints=[DomainConstraint("x", "x > 0")],
        ),
    ])


def main() -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Verifier-gated neuro-symbolic math item generator")
    parser.add_argument("--input", type=Path, help="Problem JSON file; omitted for built-in demo")
    parser.add_argument("--provider", choices=["demo", "openai", "siliconflow"], default="demo")
    parser.add_argument("--model")
    parser.add_argument("--mode", choices=[mode.value for mode in GenerationMode], default="isomorphic")
    parser.add_argument("--target-step")
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args()

    problem = Problem.from_dict(json.loads(args.input.read_text(encoding="utf-8-sig"))) if args.input else demo_problem()
    generator = demo_generator() if args.provider == "demo" else LLMGenerator(
        OpenAICompatibleClient.from_env(args.provider, args.model)
    )
    result = MathAgent(generator, max_attempts=args.max_attempts).run(
        problem, GenerationMode(args.mode), args.target_step
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

