"""Docstrings: Python's own convention (a bare string as the first
statement of a body) made to actually mean something in Sense, rather
than just being an inert expression statement whose value is computed and
discarded. Works for `tool`/`action`/`agent`/a plain function alike, with
either an ordinary `"..."` (which already tolerates an embedded literal
newline) or the newer `\"\"\"...\"\"\"` form. See values.py's
`_extract_doc`.
"""

from __future__ import annotations

from textwrap import dedent

from sense_lang.interpreter import Interpreter


def run(source: str):
    interpreter = Interpreter(stdout=lambda _s: None)
    env = interpreter.run_source(dedent(source))
    return interpreter, env


def test_tool_captures_a_triple_quoted_docstring():
    interp, env = run(
        '''
        tool greet(name: String) returns String:
            """
            Greets a user by name.
            Returns the greeting string.
            """
            return "hi " + name
        '''
    )
    tool = env.get("greet")
    assert tool.doc == "Greets a user by name.\nReturns the greeting string."


def test_tool_captures_a_plain_one_line_docstring():
    interp, env = run(
        '''
        tool greet(name: String) returns String:
            "greets a user by name"
            return "hi " + name
        '''
    )
    assert env.get("greet").doc == "greets a user by name"


def test_action_captures_a_docstring():
    interp, env = run(
        '''
        irreversible action restart_service(service: String) requires ops.remediate:
            """
            Restarts a service process.
            """
            print("restarting")
        policy:
            allow ops.remediate
        '''
    )
    assert env.get("restart_service").doc == "Restarts a service process."


def test_agent_captures_a_docstring():
    interp, env = run(
        '''
        agent researcher:
            "Investigates a topic and stores a summary in memory."
            print("working")
        '''
    )
    assert env.get("researcher").doc == "Investigates a topic and stores a summary in memory."


def test_plain_function_captures_a_docstring():
    interp, env = run(
        '''
        def square(x) -> Int:
            """Returns the square of x."""
            return x * x
        '''
    )
    assert env.get("square").doc == "Returns the square of x."


def test_no_docstring_is_nil_not_an_error():
    interp, env = run(
        '''
        tool greet(name: String) returns String:
            return "hi " + name
        '''
    )
    assert env.get("greet").doc is None


def test_a_non_string_first_statement_is_not_mistaken_for_a_docstring():
    interp, env = run(
        '''
        tool greet(name: String) returns String:
            x = 1
            return "hi " + name
        '''
    )
    assert env.get("greet").doc is None


def test_docstring_is_still_an_ordinary_harmless_statement_not_special_syntax():
    """No new grammar: the string was already a legal (if pointless)
    expression statement before this -- this only decides what capturing
    it *means*, so the body still runs exactly as it did before."""
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(
        dedent(
            '''
            tool greet(name: String) returns String:
                "greets a user by name"
                return "hi " + name
            print(greet("ada"))
            '''
        )
    )
    assert output == ["hi ada"]


def test_the_bodys_own_first_real_statement_still_runs_after_a_docstring():
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(
        dedent(
            '''
            tool double(x: Int) returns Int:
                "doubles x"
                y = x * 2
                return y
            print(double(21))
            '''
        )
    )
    assert output == ["42"]
