"""The `test "description": ...` statement (Phase 5: Developer Experience).

Each block is isolated: a failure inside is recorded, not raised, so one
failing assertion doesn't stop the rest of the file or the rest of the
suite. See Interpreter._exec_TestStmt.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseSyntaxError
from sense_lang.interpreter import Interpreter


def run(source: str) -> Interpreter:
    interp = Interpreter(stdout=lambda _: None)
    interp.run_source(dedent(source))
    return interp


def test_passing_test_is_recorded():
    interp = run('test "ok":\n    assert(1 + 1 == 2)\n')
    assert len(interp.test_results) == 1
    r = interp.test_results[0]
    assert r.description == "ok"
    assert r.passed is True
    assert r.error is None


def test_failing_assertion_is_isolated_not_raised():
    interp = run('test "bad":\n    assert(1 == 2, "nope")\n')
    r = interp.test_results[0]
    assert r.passed is False
    assert r.error == "nope"


def test_runtime_error_inside_test_is_isolated():
    interp = run('test "divide":\n    x = 1 / 0\n')
    r = interp.test_results[0]
    assert r.passed is False
    assert "division by zero" in r.error


def test_execution_continues_after_a_failing_test():
    output = []
    interp = Interpreter(stdout=output.append)
    interp.run_source(
        dedent(
            """
            test "fails":
                assert(false)
            print("still ran")
            """
        )
    )
    assert output == ["still ran"]
    assert interp.test_results[0].passed is False


def test_multiple_tests_in_one_file_each_get_their_own_result():
    interp = run(
        """
        test "a":
            assert(true)
        test "b":
            assert(false)
        test "c":
            assert(true)
        """
    )
    results = {r.description: r.passed for r in interp.test_results}
    assert results == {"a": True, "b": False, "c": True}


def test_helper_functions_are_visible_inside_test_blocks():
    interp = run(
        """
        def square(x) -> Int:
            return x * x
        test "square":
            assert(square(3) == 9)
        """
    )
    assert interp.test_results[0].passed is True


def test_inline_single_statement_test_block():
    interp = run('test "inline": assert(1 == 1)\n')
    assert interp.test_results[0].passed is True


def test_test_body_has_its_own_scope():
    with pytest.raises(Exception):
        run(
            """
            test "leaks":
                local_only = 5
            print(local_only)
            """
        )


def test_test_requires_a_string_description():
    with pytest.raises(SenseSyntaxError):
        run("test 5:\n    assert(true)\n")


def test_type_error_inside_test_is_isolated():
    interp = run(
        """
        def f(a: Int) -> Int:
            return a
        test "wrong type":
            f("nope")
        """
    )
    r = interp.test_results[0]
    assert r.passed is False
    assert "Int" in r.error
