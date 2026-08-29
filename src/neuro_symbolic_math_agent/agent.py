from __future__ import annotations

from .models import (
    AgentResult,
    Attempt,
    GenerationMode,
    Problem,
    VerificationIssue,
)
from .providers import Generator
from .reflection import ReflectionEngine
from .verifier import SymbolicVerifier


class MathAgent:
    def __init__(
        self,
        generator: Generator,
        verifier: SymbolicVerifier | None = None,
        reflection_engine: ReflectionEngine | None = None,
        max_attempts: int = 5,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.generator = generator
        self.verifier = verifier or SymbolicVerifier()
        self.reflection_engine = reflection_engine or ReflectionEngine()
        self.max_attempts = max_attempts

    def run(
        self,
        problem: Problem,
        mode: GenerationMode = GenerationMode.ISOMORPHIC,
        target_step_id: str | None = None,
    ) -> AgentResult:
        self._validate_problem(problem)
        if mode == GenerationMode.SCAFFOLD:
            if not target_step_id:
                raise ValueError("target_step_id is required in scaffold mode")
            if target_step_id not in {step.step_id for step in problem.steps}:
                raise ValueError(f"Unknown target step: {target_step_id}")

        attempts: list[Attempt] = []
        reflection: str | None = None
        for number in range(1, self.max_attempts + 1):
            try:
                candidate = self.generator.generate(problem, mode, target_step_id, reflection)
            except Exception as exc:
                attempts.append(Attempt(number, None, None, generation_error=str(exc)))
                reflection = "Generation failed. Return a complete candidate matching the required JSON schema."
                continue

            result = self.verifier.verify(candidate, problem, mode)
            if mode == GenerationMode.SCAFFOLD and candidate.target_step_id != target_step_id:
                result.accepted = False
                result.issues.append(VerificationIssue(
                    "TARGET_STEP_MISMATCH",
                    f"Candidate targets {candidate.target_step_id!r}, expected {target_step_id!r}.",
                ))
            if result.accepted:
                attempts.append(Attempt(number, candidate, result))
                return AgentResult(True, candidate, result, attempts)

            reflection = self.reflection_engine.build(result)
            attempts.append(Attempt(number, candidate, result, reflection=reflection))

        reason = f"Verifier gate abstained after {self.max_attempts} unsuccessful attempts."
        return AgentResult(False, None, None, attempts, reason)

    @staticmethod
    def _validate_problem(problem: Problem) -> None:
        ids = [step.step_id for step in problem.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Step IDs must be unique")
        known = set(ids)
        for step in problem.steps:
            unknown = set(step.dependencies) - known
            if unknown:
                raise ValueError(f"Step {step.step_id} has unknown dependencies: {sorted(unknown)}")
        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {step.step_id: step.dependencies for step in problem.steps}

        def visit(node: str) -> None:
            if node in visiting:
                raise ValueError("Step dependency graph must be acyclic")
            if node in visited:
                return
            visiting.add(node)
            for dependency in graph[node]:
                visit(dependency)
            visiting.remove(node)
            visited.add(node)

        for node in graph:
            visit(node)
