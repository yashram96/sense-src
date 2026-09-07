"""`tool` — a capability-checked function with no prepare/verify/commit
lifecycle. Distinct from `action`: calling a tool runs its
body immediately; the capability, if any, is checked once at the call site
against the caller's policy scope.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SensePolicyError, SenseRuntimeError, SenseTypeError
from sense_lang.interpreter import Interpreter


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


def test_tool_call_runs_the_body_immediately_no_lifecycle():
    out = run(
        """
        tool search(query: String) returns String:
            return "results for " + query
        print(search("sense"))
        """
    )
    assert out == ["results for sense"]


def test_tool_with_no_requires_clause_is_unrestricted():
    out = run(
        """
        tool add(a: Int, b: Int) returns Int:
            return a + b
        policy:
            deny anything.at.all
        print(add(1, 2))
        """
    )
    assert out == ["3"]


def test_tool_call_is_capability_gated_and_denied():
    with pytest.raises(SensePolicyError):
        run(
            """
            tool search(query: String) requires web.search:
                return query
            policy:
                deny web.search
            search("x")
            """
        )


def test_tool_call_succeeds_when_capability_is_allowed():
    out = run(
        """
        tool search(query: String) returns String requires web.search:
            return "found: " + query
        policy:
            allow web.search
        print(search("sense"))
        """
    )
    assert out == ["found: sense"]


def test_tool_argument_type_is_checked():
    with pytest.raises(SenseTypeError):
        run(
            """
            tool square(x: Int) returns Int:
                return x * x
            square("not an int")
            """
        )


def test_tool_return_type_is_checked():
    with pytest.raises(SenseTypeError):
        run(
            """
            tool bad() returns Int:
                return "not an int"
            bad()
            """
        )


def test_tool_arity_mismatch_is_a_runtime_error():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            tool one(x: Int) returns Int:
                return x
            one(1, 2)
            """
        )


def test_type_of_a_tool():
    out = run(
        """
        tool t():
            return nil
        print(type_of(t))
        """
    )
    assert out == ["Tool"]


def test_tool_calls_are_audited():
    out = run(
        """
        tool search(query: String) requires web.search:
            return query
        policy:
            allow web.search
        search("x")
        for line in audit_log():
            print(line)
        """
    )
    assert any("search" in line and "called" in line for line in out)
