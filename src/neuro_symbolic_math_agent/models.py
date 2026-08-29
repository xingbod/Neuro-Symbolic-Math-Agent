from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]



def _equation_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("expression", "equation", "expr", "text"):
            if value.get(key) is not None:
                return str(value[key])
    return str(value)

def _variable_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("name", "symbol", "variable", "id"):
            if value.get(key):
                return str(value[key])
    raise ValueError(f"Cannot determine variable name from {value!r}")


def _constraint(value: Any) -> "DomainConstraint":
    if isinstance(value, str):
        expression = value
        names = [
            name for name in __import__("re").findall(r"[A-Za-z_]\w*", expression)
            if name not in {"Ne", "Eq", "And", "Or", "sqrt", "Abs", "True", "False"}
        ]
        if not names:
            raise ValueError(f"Cannot infer constrained variable from {value!r}")
        return DomainConstraint(names[0], expression, True)
    if not isinstance(value, dict):
        raise ValueError(f"Domain constraint must be an object or string, got {value!r}")
    expression = value.get("expression", value.get("constraint", value.get("condition")))
    variable = value.get("variable", value.get("name", value.get("symbol")))
    if expression is None or variable is None:
        raise ValueError(f"Domain constraint needs variable and expression: {value!r}")
    return DomainConstraint(
        variable=_variable_name(variable),
        expression=str(expression),
        require_all_solutions=bool(value.get("require_all_solutions", True)),
    )


class GenerationMode(str, Enum):
    ISOMORPHIC = "isomorphic"
    SCAFFOLD = "scaffold"


@dataclass(slots=True)
class Step:
    step_id: str
    name: str
    description: str
    equations: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> "Step":
        step_id = data.get("step_id", data.get("id", f"s{index + 1}"))
        dependencies = [
            str(item.get("step_id", item.get("id"))) if isinstance(item, dict) else str(item)
            for item in _list(data.get("dependencies", data.get("deps", [])))
        ]
        return cls(
            step_id=str(step_id),
            name=str(data.get("name", f"step_{index + 1}")),
            description=str(data.get("description", data.get("text", ""))),
            equations=[_equation_text(item) for item in _list(data.get("equations", []))],
            dependencies=dependencies,
            concepts=[str(item) for item in _list(data.get("concepts", []))],
        )


@dataclass(slots=True)
class DomainConstraint:
    variable: str
    expression: str
    require_all_solutions: bool = True


@dataclass(slots=True)
class Problem:
    problem_id: str
    text: str
    variables: list[str]
    equations: list[str]
    steps: list[Step]
    concepts: list[str] = field(default_factory=list)
    domain_constraints: list[DomainConstraint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Problem":
        return cls(
            problem_id=str(data.get("problem_id", "problem")),
            text=str(data.get("text", data.get("problem", ""))),
            variables=[_variable_name(item) for item in _list(data.get("variables"))],
            equations=[_equation_text(item) for item in _list(data.get("equations"))],
            steps=[Step.from_dict(step, index) for index, step in enumerate(_list(data.get("steps", [])))],
            concepts=[str(item) for item in _list(data.get("concepts", []))],
            domain_constraints=[_constraint(item) for item in _list(data.get("domain_constraints", []))],
        )


@dataclass(slots=True)
class Candidate:
    question: str
    variables: list[str]
    equations: list[str]
    answer: str = ""
    target_step_id: str | None = None
    domain_constraints: list[DomainConstraint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            question=str(data.get("question", data.get("text", ""))),
            variables=[_variable_name(item) for item in _list(data.get("variables"))],
            equations=[_equation_text(item) for item in _list(data.get("equations"))],
            answer=str(data.get("answer", "")),
            target_step_id=str(data["target_step_id"]) if data.get("target_step_id") is not None else None,
            domain_constraints=[_constraint(item) for item in _list(data.get("domain_constraints", []))],
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class VerificationIssue:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VerificationResult:
    accepted: bool
    solutions: list[dict[str, str]] = field(default_factory=list)
    cvi: float = 0.0
    isomorphism_score: float = 0.0
    issues: list[VerificationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Attempt:
    number: int
    candidate: Candidate | None
    verification: VerificationResult | None
    reflection: str | None = None
    generation_error: str | None = None


@dataclass(slots=True)
class AgentResult:
    delivered: bool
    candidate: Candidate | None
    verification: VerificationResult | None
    attempts: list[Attempt]
    abstention_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



