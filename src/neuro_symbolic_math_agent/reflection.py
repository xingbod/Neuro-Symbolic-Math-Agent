from __future__ import annotations

from .models import VerificationResult


class ReflectionEngine:
    DIRECTIVES = {
        "PARSE_ERROR": "Return valid SymPy-compatible equations and only declared variable names.",
        "SOLVER_ERROR": "Simplify the equation system and ensure it is symbolically solvable.",
        "NO_SOLUTION_ERROR": "Construct the answer values first, then derive coefficients so a finite solution exists.",
        "NON_REAL_SOLUTION_ERROR": "Ensure all intended roots are real; for quadratics enforce discriminant >= 0.",
        "DOMAIN_VIOLATION_ERROR": "Regenerate values so every required semantic domain constraint is satisfied.",
        "UGLY_SOLUTIONS_ERROR": "Use small integer roots in [-20, 20] or simple fractions with denominator <= 12.",
        "NON_ISOMORPHIC_ERROR": "Preserve variable count, equation count, polynomial degree, and operation structure.",
        "ORIGINAL_PARSE_ERROR": "Keep the candidate valid; the source item needs manual schema correction.",
    }

    def build(self, result: VerificationResult) -> str:
        lines = ["The verifier rejected the previous candidate. Correct every issue:"]
        for issue in result.issues:
            directive = self.DIRECTIVES.get(issue.code, "Correct the reported verifier failure.")
            lines.append(f"- {issue.code}: {issue.message} Directive: {directive}")
        lines.append("Return one complete JSON candidate only; do not explain the correction.")
        return "\n".join(lines)
