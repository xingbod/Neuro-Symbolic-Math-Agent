import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest

from neuro_symbolic_math_agent.agent import MathAgent
from neuro_symbolic_math_agent.models import Candidate, DomainConstraint, GenerationMode, Problem, Step
from neuro_symbolic_math_agent.providers import SequenceGenerator
from neuro_symbolic_math_agent.verifier import SymbolicVerifier


def source_problem() -> Problem:
    return Problem(
        "p1", "Solve x²-5x+6=0", ["x"], ["x**2-5*x+6=0"],
        [Step("s1", "factor", "Factor"), Step("s2", "solve", "Solve", dependencies=["s1"])],
        ["quadratics"], [DomainConstraint("x", "x > 0")],
    )


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = SymbolicVerifier()
        self.problem = source_problem()

    def test_accepts_clean_isomorphic_quadratic(self):
        candidate = Candidate("Solve", ["x"], ["x**2-7*x+12=0"], domain_constraints=[DomainConstraint("x", "x > 0")])
        result = self.verifier.verify(candidate, self.problem)
        self.assertTrue(result.accepted)
        self.assertEqual(result.cvi, 1.0)
        self.assertGreaterEqual(result.isomorphism_score, 0.7)
        self.assertEqual(result.solutions, [{"x": "3"}, {"x": "4"}])

    def test_rejects_non_real_roots(self):
        candidate = Candidate("Solve", ["x"], ["x**2+x+1=0"])
        result = self.verifier.verify(candidate, self.problem)
        self.assertFalse(result.accepted)
        self.assertIn("NO_SOLUTION_ERROR", {issue.code for issue in result.issues})

    def test_rejects_domain_violation(self):
        candidate = Candidate("Solve", ["x"], ["x**2-x-2=0"], domain_constraints=[DomainConstraint("x", "x > 0")])
        result = self.verifier.verify(candidate, self.problem)
        self.assertFalse(result.accepted)
        self.assertIn("DOMAIN_VIOLATION_ERROR", {issue.code for issue in result.issues})

    def test_rationalizes_finite_decimals(self):
        candidate = Candidate("Solve", ["x"], ["x=1.2"], domain_constraints=[DomainConstraint("x", "x > 0")])
        result = self.verifier.verify(candidate)
        self.assertTrue(result.accepted)
        self.assertEqual(result.solutions, [{"x": "6/5"}])

    def test_supports_not_equal_domain_constraint(self):
        candidate = Candidate("Solve", ["x"], ["x=2"], domain_constraints=[DomainConstraint("x", "x != 0")])
        self.assertTrue(self.verifier.verify(candidate).accepted)

    def test_rejects_unknown_symbols(self):
        candidate = Candidate("Solve", ["x"], ["x+y=2"])
        result = self.verifier.verify(candidate, self.problem)
        self.assertFalse(result.accepted)
        self.assertEqual(result.issues[0].code, "PARSE_ERROR")


class AgentTests(unittest.TestCase):
    def test_reflects_then_delivers(self):
        bad = Candidate("bad", ["x"], ["x**2+x+1=0"])
        good = Candidate("good", ["x"], ["x**2-7*x+12=0"], domain_constraints=[DomainConstraint("x", "x > 0")])
        generator = SequenceGenerator([bad, good])
        result = MathAgent(generator, max_attempts=3).run(source_problem())
        self.assertTrue(result.delivered)
        self.assertEqual(len(result.attempts), 2)
        self.assertIn("NO_SOLUTION_ERROR", generator.reflections[1])

    def test_abstains_instead_of_delivering_invalid_item(self):
        bad = Candidate("bad", ["x"], ["x**2+x+1=0"])
        result = MathAgent(SequenceGenerator([bad]), max_attempts=2).run(source_problem())
        self.assertFalse(result.delivered)
        self.assertIsNone(result.candidate)
        self.assertEqual(len(result.attempts), 2)

    def test_scaffold_requires_matching_step(self):
        candidate = Candidate("scaffold", ["x"], ["x-2=0"], target_step_id="s2")
        result = MathAgent(SequenceGenerator([candidate]), max_attempts=1).run(
            source_problem(), GenerationMode.SCAFFOLD, "s1"
        )
        self.assertFalse(result.delivered)
        self.assertIn("TARGET_STEP_MISMATCH", {i.code for i in result.attempts[0].verification.issues})


class ModelParsingTests(unittest.TestCase):
    def test_problem_accepts_structured_variables(self):
        problem = Problem.from_dict({
            "text": "Solve", "variables": [{"name": "x", "domain": "real"}],
            "equations": [{"expression": "x=2", "purpose": "given"}], "steps": [{"id": "s1", "text": "solve", "dependencies": []}],
            "domain_constraints": ["Ne(x, 0)"],
        })
        self.assertEqual(problem.variables, ["x"])
        self.assertEqual(problem.steps[0].step_id, "s1")
        self.assertEqual(problem.domain_constraints[0].expression, "Ne(x, 0)")


if __name__ == "__main__":
    unittest.main()
