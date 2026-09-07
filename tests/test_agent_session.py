"""Phase 3 core slice: session (scoped delegation) and agent (identity,
persistent state, partial lifecycle)."""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseError, SenseRuntimeError
from sense_lang.interpreter import Interpreter
from sense_lang.lexer import tokenize
from sense_lang.parser import parse


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


# -- session ------------------------------------------------------------------


def test_session_runs_its_body():
    out = run(
        """
        session s:
            x = 5
            print(x)
        print("after")
        """
    )
    assert out == ["5", "after"]


def test_session_variables_do_not_leak_out():
    with pytest.raises(SenseError):
        run(
            """
            session s:
                x = 5
            print(x)
            """
        )


def test_session_scopes_delegation_without_leaking():
    out = run(
        """
        model outer_model = inference("mock", "outer", response_template: "outer: {prompt}")
        set delegation = outer_model

        session s:
            model inner_model = inference("mock", "inner", response_template: "inner: {prompt}")
            set delegation = inner_model
            print(ask("q").value)

        print(ask("q").value)
        """
    )
    assert out == ["inner: q", "outer: q"]


def test_session_inherits_outer_delegation_by_default():
    out = run(
        """
        model m = inference("mock", "shared", response_template: "shared: {prompt}")
        set delegation = m
        session s:
            print(ask("q").value)
        """
    )
    assert out == ["shared: q"]


# -- agent ----------------------------------------------------------------------


def test_agent_decl_creates_but_does_not_run():
    out = run(
        """
        agent a:
            print("should not print yet")
        print(a.status)
        """
    )
    assert out == ["created"]


def test_starting_agent_runs_body_and_completes():
    out = run(
        """
        agent a:
            print("running")
        print(a.status)
        start(a)
        print(a.status)
        """
    )
    assert out == ["created", "running", "completed"]


def test_agent_state_is_inspectable_after_completion():
    out = run(
        """
        agent researcher:
            goal: String = "find the best database"
            answer = "postgres"
        start(researcher)
        print(researcher.goal)
        print(researcher.answer)
        """
    )
    assert out == ["find the best database", "postgres"]


def test_agent_start_returns_the_agent():
    out = run(
        """
        agent a:
            x = 1
        result = start(a)
        print(result.status)
        print(type_of(result))
        """
    )
    assert out == ["completed", "Agent"]


def test_starting_agent_twice_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            agent a:
                x = 1
            start(a)
            start(a)
            """
        )


def test_agent_failure_sets_status_failed_and_reraises():
    # Two run_source() calls each get a fresh module scope off the same
    # globals, so drive the second statement directly against the first
    # call's env to keep `broken` visible.
    interp = Interpreter(stdout=lambda _: None)
    env = interp.run_source("agent broken:\n    x = 1 / 0\n")
    agent = env.get("broken")
    assert agent.status == "created"

    start_stmt = parse(tokenize("start(broken)\n")).statements[0]
    with pytest.raises(SenseError):
        interp._execute(start_stmt, env, None)  # noqa: SLF001 - low-level, see comment above
    assert agent.status == "failed"


def test_start_on_non_agent_raises():
    with pytest.raises(SenseRuntimeError):
        run("x = 5\nstart(x)\n")


def test_agent_has_its_own_delegation_scope():
    out = run(
        """
        model outer_model = inference("mock", "outer", response_template: "outer: {prompt}")
        set delegation = outer_model

        agent a:
            model inner_model = inference("mock", "inner", response_template: "inner: {prompt}")
            set delegation = inner_model
            print(ask("q").value)

        start(a)
        print(ask("q").value)
        """
    )
    assert out == ["inner: q", "outer: q"]


def test_agent_env_is_lexically_nested_like_a_closure():
    out = run(
        """
        x = 1
        agent a:
            x = 2
            print(x)
        start(a)
        print(x)
        """
    )
    # a.env is env.child() at declaration time (lexical, like a function
    # closure) — plain assignment walks up and mutates the existing outer
    # `x`, same "one assignment rule" as everywhere else in the language.
    # Use `local x = 2` inside the agent body if isolation is wanted.
    assert out == ["2", "2"]


def test_type_of_agent():
    out = run("agent a:\n    x = 1\nprint(type_of(a))\n")
    assert out == ["Agent"]
