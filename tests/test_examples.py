"""Smoke-test every example program end-to-end via the interpreter."""

from pathlib import Path

import pytest

from sense_lang.errors import SensePolicyError
from sense_lang.interpreter import Interpreter

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

# rollback_approval_workflow_failure.sns is deliberately excluded from the
# "runs without error" sweep below -- ending in an uncaught SensePolicyError
# is its entire point (see its own header comment). It gets its own test,
# right after, asserting exactly that failure instead of a passing run.
_INTENTIONALLY_FAILING = {"rollback_approval_workflow_failure.sns"}
EXAMPLE_FILES = sorted(p for p in EXAMPLES_DIR.rglob("*.sns") if p.name not in _INTENTIONALLY_FAILING)


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: str(p.relative_to(EXAMPLES_DIR)))
def test_example_runs_without_error(path: Path):
    interp = Interpreter(stdout=lambda _: None)
    interp.run_file(str(path))


def test_rollback_approval_workflow_failure_example_fails_as_documented():
    # The companion to rollback_approval_workflow.sns: verify() succeeds
    # while the capability is granted, then the capability is revoked
    # before commit() -- proving commit() really does re-check
    # independently, not just claiming it in prose.
    interp = Interpreter(stdout=lambda _: None)
    with pytest.raises(SensePolicyError, match="deploy.rollback"):
        interp.run_file(str(EXAMPLES_DIR / "rollback_approval_workflow_failure.sns"))


def test_fib_example_produces_expected_sequence():
    output: list[str] = []
    interp = Interpreter(stdout=output.append)
    interp.run_file(str(EXAMPLES_DIR / "fib.sns"))
    assert output == ["0", "1", "1", "2", "3", "5", "8", "13", "21", "34"]


def test_modules_example_output():
    output: list[str] = []
    interp = Interpreter(stdout=output.append)
    interp.run_file(str(EXAMPLES_DIR / "modules" / "main.sns"))
    assert output == ["25", "5", "3.14159"]
