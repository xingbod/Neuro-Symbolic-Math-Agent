from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sympy as sp
from sympy.core.relational import Relational

from .models import (
    Candidate,
    DomainConstraint,
    GenerationMode,
    Problem,
    VerificationIssue,
    VerificationResult,
)


@dataclass(slots=True)
class VerifierConfig:
    min_cvi: float = 0.70
    min_isomorphism: float = 0.70


class ParseError(ValueError):
    pass


class SymbolicVerifier:
    """Four-pass gate: parse, solve/realness, domain/CVI, structural similarity."""

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self.config = config or VerifierConfig()

    def verify(
        self,
        candidate: Candidate,
        original: Problem | None = None,
        mode: GenerationMode = GenerationMode.ISOMORPHIC,
    ) -> VerificationResult:
        issues: list[VerificationIssue] = []
        try:
            symbols = self._symbols(candidate.variables)
            equations = self._parse_equations(candidate.equations, symbols)
        except ParseError as exc:
            return VerificationResult(False, issues=[VerificationIssue("PARSE_ERROR", str(exc))])

        try:
            raw_solutions = sp.solve(equations, list(symbols.values()), dict=True)
        except Exception as exc:
            return VerificationResult(False, issues=[VerificationIssue("SOLVER_ERROR", str(exc))])

        if not raw_solutions or any(any(symbol not in solution for symbol in symbols.values()) for solution in raw_solutions):
            return VerificationResult(
                False,
                issues=[VerificationIssue("NO_SOLUTION_ERROR", "No finite, fully determined solution was found.")],
            )

        non_real = [value for solution in raw_solutions for value in solution.values() if not self._is_real(value)]
        if non_real:
            issues.append(VerificationIssue(
                "NON_REAL_SOLUTION_ERROR",
                "At least one solution is non-real or cannot be proven real.",
                {"values": [str(value) for value in non_real]},
            ))

        constraints = candidate.domain_constraints or (original.domain_constraints if original else [])
        try:
            domain_failures = self._domain_failures(raw_solutions, constraints, symbols)
        except ParseError as exc:
            domain_failures = []
            issues.append(VerificationIssue("DOMAIN_PARSE_ERROR", str(exc)))
        if domain_failures:
            issues.append(VerificationIssue(
                "DOMAIN_VIOLATION_ERROR",
                "One or more solutions violate semantic domain constraints.",
                {"failures": domain_failures},
            ))

        values = [value for solution in raw_solutions for value in solution.values()]
        cvi = self.clean_value_index(values)
        if cvi < self.config.min_cvi:
            issues.append(VerificationIssue(
                "UGLY_SOLUTIONS_ERROR",
                f"CVI {cvi:.3f} is below threshold {self.config.min_cvi:.3f}.",
                {"cvi": cvi, "threshold": self.config.min_cvi},
            ))

        iso_score = 1.0
        if original is not None and mode == GenerationMode.ISOMORPHIC:
            try:
                original_symbols = self._symbols(original.variables)
                original_equations = self._parse_equations(original.equations, original_symbols)
                iso_score = self.isomorphism_score(equations, symbols, original_equations, original_symbols)
            except ParseError as exc:
                issues.append(VerificationIssue("ORIGINAL_PARSE_ERROR", str(exc)))
                iso_score = 0.0
            if iso_score < self.config.min_isomorphism:
                issues.append(VerificationIssue(
                    "NON_ISOMORPHIC_ERROR",
                    f"Structural score {iso_score:.3f} is below threshold {self.config.min_isomorphism:.3f}.",
                    {"score": iso_score, "threshold": self.config.min_isomorphism},
                ))

        rendered = [
            {str(symbol): str(sp.simplify(value)) for symbol, value in solution.items()}
            for solution in raw_solutions
        ]
        return VerificationResult(not issues, rendered, cvi, iso_score, issues)

    @staticmethod
    def _symbols(names: Iterable[str]) -> dict[str, sp.Symbol]:
        clean = list(names)
        if not clean or len(clean) != len(set(clean)):
            raise ParseError("Variables must be a non-empty unique list.")
        if any(not name.isidentifier() for name in clean):
            raise ParseError("Variable names must be valid identifiers.")
        return {name: sp.Symbol(name, real=True) for name in clean}

    @staticmethod
    def _safe_expr(text: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
        try:
            expr = sp.S(sp.sympify(text.strip(), locals=symbols, evaluate=True, rational=True))
        except Exception as exc:
            raise ParseError(f"Cannot parse expression {text!r}: {exc}") from exc
        unknown = expr.free_symbols - set(symbols.values())
        if unknown:
            raise ParseError(f"Unknown symbols in {text!r}: {', '.join(map(str, unknown))}")
        return expr

    def _parse_constraint(self, text: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
        raw = text.strip()
        for operator, constructor in (("!=", sp.Ne), (">=", sp.Ge), ("<=", sp.Le), ("==", sp.Eq)):
            if operator in raw:
                left, right = raw.split(operator, 1)
                return constructor(self._safe_expr(left, symbols), self._safe_expr(right, symbols))
        if raw.count("=") == 1:
            left, right = raw.split("=", 1)
            return sp.Eq(self._safe_expr(left, symbols), self._safe_expr(right, symbols))
        return self._safe_expr(raw, symbols)
    def _parse_equations(self, equations: list[str], symbols: dict[str, sp.Symbol]) -> list[sp.Expr]:
        if not equations:
            raise ParseError("At least one equation is required.")
        parsed: list[sp.Expr] = []
        for equation in equations:
            if equation.count("=") > 1:
                raise ParseError(f"Equation has multiple '=' signs: {equation!r}")
            if "=" in equation:
                left, right = equation.split("=", 1)
                parsed.append(self._safe_expr(left, symbols) - self._safe_expr(right, symbols))
            else:
                parsed.append(self._safe_expr(equation, symbols))
        return parsed

    @staticmethod
    def _is_real(value: sp.Expr) -> bool:
        simplified = sp.simplify(value)
        if simplified.is_real is True:
            return True
        if simplified.is_real is False:
            return False
        try:
            return abs(complex(sp.N(simplified)).imag) < 1e-12
        except Exception:
            return False

    def _domain_failures(
        self,
        solutions: list[dict[sp.Symbol, sp.Expr]],
        constraints: list[DomainConstraint],
        symbols: dict[str, sp.Symbol],
    ) -> list[dict[str, str | int]]:
        failures: list[dict[str, str | int]] = []
        for constraint in constraints:
            if constraint.variable not in symbols:
                failures.append({"constraint": constraint.expression, "reason": "unknown variable"})
                continue
            relation = self._parse_constraint(constraint.expression, symbols)
            if not isinstance(relation, (Relational, sp.logic.boolalg.Boolean)):
                raise ParseError(f"Domain constraint must be relational: {constraint.expression!r}")
            outcomes: list[bool] = []
            for index, solution in enumerate(solutions):
                evaluated = sp.simplify(relation.subs(solution))
                passed = evaluated is sp.true or evaluated == True  # noqa: E712
                outcomes.append(passed)
                if constraint.require_all_solutions and not passed:
                    failures.append({
                        "constraint": constraint.expression,
                        "solution_index": index,
                        "reason": str(evaluated),
                    })
            if not constraint.require_all_solutions and not any(outcomes):
                failures.append({"constraint": constraint.expression, "reason": "no solution satisfies constraint"})
        return failures

    @classmethod
    def cleanliness_weight(cls, value: sp.Expr) -> float:
        value = sp.simplify(value)
        if not cls._is_real(value) or value.has(sp.Float):
            return 0.0
        if value.is_Integer:
            magnitude = abs(int(value))
            if magnitude <= 100:
                return 1.0
            if magnitude <= 1000:
                return 0.90
            return 0.40
        if value.is_Rational:
            numerator, denominator = abs(int(value.p)), abs(int(value.q))
            if denominator <= 12 and numerator <= 100:
                return 0.95
            if denominator <= 50:
                return 0.85
            return max(0.10, 0.85 * 50 / denominator)
        if value.has(sp.sqrt) or any(power.exp == sp.Rational(1, 2) for power in value.atoms(sp.Pow)):
            return 0.75
        if value.is_algebraic:
            return 0.50
        return 0.0

    @classmethod
    def clean_value_index(cls, values: list[sp.Expr]) -> float:
        if not values:
            return 0.0
        return sum(cls.cleanliness_weight(value) for value in values) / len(values)

    @staticmethod
    def isomorphism_score(
        candidate_equations: list[sp.Expr],
        candidate_symbols: dict[str, sp.Symbol],
        original_equations: list[sp.Expr],
        original_symbols: dict[str, sp.Symbol],
    ) -> float:
        score = 1.0
        if len(candidate_symbols) != len(original_symbols):
            score -= 0.20
        if len(candidate_equations) != len(original_equations):
            score -= min(0.30, 0.15 * abs(len(candidate_equations) - len(original_equations)))

        def signature(expressions: list[sp.Expr], symbols: dict[str, sp.Symbol]) -> tuple[int, int]:
            degrees: list[int] = []
            for expression in expressions:
                try:
                    degrees.append(int(sp.Poly(expression, *symbols.values()).total_degree()))
                except sp.PolynomialError:
                    degrees.append(5)
            return max(degrees, default=0), sum(int(sp.count_ops(expression)) for expression in expressions)

        candidate_degree, candidate_ops = signature(candidate_equations, candidate_symbols)
        original_degree, original_ops = signature(original_equations, original_symbols)
        score -= 0.20 * abs(candidate_degree - original_degree)
        score -= 0.20 * abs(candidate_ops - original_ops) / max(1, candidate_ops, original_ops)
        return max(0.0, min(1.0, score))


