"""Agent pause/resume: cooperative hand-off on an OS thread per running
agent (see SenseAgent's docstring in values.py). Exactly one of the
caller or the agent is ever actually executing."""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseError, SenseRuntimeError
from sense_lang.interpreter import Interpreter


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


def test_pause_suspends_and_start_returns_early():
    out = run(
        """
        agent a:
            print("before")
            pause()
            print("after")
        start(a)
        print(a.status)
        """
    )
    assert out == ["before", "paused"]


def test_resume_continues_exactly_where_it_paused():
    out = run(
        """
        agent a:
            print("before")
            pause()
            print("after")
        start(a)
        resume(a)
        print(a.status)
        """
    )
    assert out == ["before", "after", "completed"]


def test_multiple_pause_points_in_sequence():
    out = run(
        """
        agent a:
            print(1)
            pause()
            print(2)
            pause()
            print(3)
        start(a)
        resume(a)
        resume(a)
        print(a.status)
        """
    )
    assert out == ["1", "2", "3", "completed"]


def test_local_state_survives_across_pause():
    out = run(
        """
        agent a:
            x = 1
            pause()
            x = x + 1
            pause()
            print(x)
        start(a)
        resume(a)
        resume(a)
        """
    )
    assert out == ["2"]


def test_pause_with_reason_is_readable():
    out = run(
        """
        agent a:
            pause("waiting for approval")
        start(a)
        print(a.pause_reason)
        """
    )
    assert out == ["waiting for approval"]


def test_pause_reason_defaults_to_nil():
    out = run(
        """
        agent a:
            pause()
        start(a)
        print(a.pause_reason)
        """
    )
    assert out == ["nil"]


def test_pause_outside_any_agent_raises():
    with pytest.raises(SenseRuntimeError):
        run("pause()\n")


def test_pause_with_too_many_args_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            agent a:
                pause("one", "two")
            start(a)
            """
        )


def test_resume_on_non_agent_raises():
    with pytest.raises(SenseRuntimeError):
        run("x = 5\nresume(x)\n")


def test_resume_on_never_started_agent_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            agent a:
                print(1)
            resume(a)
            """
        )


def test_resume_on_completed_agent_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            agent a:
                print(1)
            start(a)
            resume(a)
            """
        )


def test_resume_on_already_running_or_failed_agent_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            agent a:
                x = 1 / 0
            start(a)
            resume(a)
            """
        )


def test_failure_after_pause_surfaces_on_resume_caller():
    out: list[str] = []
    interp = Interpreter(stdout=out.append)
    env = interp.run_source(
        dedent(
            """
            agent a:
                pause()
                x = 1 / 0
            start(a)
            print(a.status)
            """
        )
    )
    assert out == ["paused"]
    agent = env.get("a")
    from sense_lang.lexer import tokenize
    from sense_lang.parser import parse

    resume_stmt = parse(tokenize("resume(a)\n")).statements[0]
    with pytest.raises(SenseError):
        interp._execute(resume_stmt, env, None)  # noqa: SLF001 - see comment in test_agent_session.py
    assert agent.status == "failed"


def test_never_resumed_agent_does_not_hang_process():
    # The agent's thread stays blocked forever on its own resume event; the
    # thread is a daemon thread specifically so this is fine to leave.
    out = run(
        """
        agent a:
            pause()
            print("never reached")
        start(a)
        print("done")
        """
    )
    assert out == ["done"]


def test_delegation_set_before_pause_is_still_visible_after_resume():
    out = run(
        """
        model m = inference("mock", "x", response_template: "answer: {prompt}")
        agent a:
            set delegation = m
            pause()
            print(ask("q").value)
        start(a)
        resume(a)
        """
    )
    assert out == ["answer: q"]
