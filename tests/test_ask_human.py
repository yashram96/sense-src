"""ask_human(question): a human-intervention primitive, built directly on
pause()/resume() carrying a value across the suspend boundary."""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseRuntimeError
from sense_lang.interpreter import Interpreter


def run(source: str, stdin=None) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append, stdin=stdin).run_source(dedent(source))
    return output


# -- pause()/resume() now carry a value -----------------------------------------


def test_resume_with_value_becomes_pause_return_value():
    out = run(
        """
        agent a:
            v = pause()
            print(v)
        start(a)
        resume(a, "hello")
        """
    )
    assert out == ["hello"]


def test_resume_without_value_defaults_to_nil():
    out = run(
        """
        agent a:
            v = pause()
            print(type_of(v))
        start(a)
        resume(a)
        """
    )
    assert out == ["Nil"]


def test_resume_value_does_not_leak_to_next_pause():
    out = run(
        """
        agent a:
            v1 = pause()
            print(v1)
            v2 = pause()
            print(v2)
        start(a)
        resume(a, "first")
        resume(a)
        """
    )
    assert out == ["first", "nil"]


# -- ask_human inside an agent: built on pause/resume ----------------------------


def test_ask_human_inside_agent_pauses_with_question_as_reason():
    out = run(
        """
        agent a:
            answer = ask_human("continue?")
            print(answer)
        start(a)
        """
    )
    assert out == []  # paused before the print


def test_ask_human_pause_reason_is_the_question():
    output: list[str] = []
    interp = Interpreter(stdout=output.append)
    env = interp.run_source(
        dedent(
            """
            agent a:
                ask_human("continue?")
            start(a)
            """
        )
    )
    agent = env.get("a")
    assert agent.status == "paused"
    assert agent.pause_reason == "continue?"


def test_ask_human_answer_flows_back_via_resume():
    out = run(
        """
        agent a:
            answer = ask_human("approve?")
            print("got: " + answer)
        start(a)
        resume(a, "yes")
        """
    )
    assert out == ["got: yes"]


def test_ask_human_requires_string_question():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            agent a:
                ask_human(42)
            start(a)
            """
        )


# -- ask_human outside any agent: falls back to real stdin ----------------------


def test_ask_human_outside_agent_uses_stdin_fallback():
    out = run(
        'name = ask_human("what is your name?")\nprint("hello, " + name)\n',
        stdin=lambda prompt: "Ada",
    )
    assert out == ["what is your name?", "hello, Ada"]


def test_ask_human_stdin_fallback_receives_a_prompt_string():
    seen_prompts = []

    def fake_stdin(prompt):
        seen_prompts.append(prompt)
        return "ok"

    run('ask_human("q")\n', stdin=fake_stdin)
    assert seen_prompts == ["> "]
