from textwrap import dedent

import pytest

from sense_lang.errors import SenseRuntimeError, SenseTypeError, SenseNameError, SenseImportError
from sense_lang.interpreter import Interpreter


def run(source: str, file_path: str | None = None) -> list[str]:
    # Multi-line triple-quoted sources below inherit this test file's own
    # indentation; Sense's indentation is significant, so dedent first.
    output: list[str] = []
    interp = Interpreter(stdout=output.append)
    interp.run_source(dedent(source), file_path=file_path)
    return output


def test_arithmetic_and_precedence():
    assert run("print(1 + 2 * 3)\n") == ["7"]
    assert run("print((1 + 2) * 3)\n") == ["9"]
    assert run("print(7 % 3)\n") == ["1"]
    assert run("print(7 / 2)\n") == ["3.5"]


def test_string_concat_requires_strings():
    assert run('print("a" + "b")\n') == ["ab"]
    with pytest.raises(SenseRuntimeError):
        run('print("a" + 1)\n')


def test_variables_and_reassignment():
    out = run("x = 1\nx = x + 1\nprint(x)\n")
    assert out == ["2"]


def test_typed_decl_and_untyped_assign_are_interchangeable():
    out = run("x: Int = 1\nx = x + 1\nprint(x)\n")
    assert out == ["2"]


def test_if_else():
    out = run('x = 5\nif x > 10:\n    print("big")\nelse:\n    print("small")\n')
    assert out == ["small"]


def test_inline_if_block():
    out = run('x = 5\nif x > 10: print("big")\nelse: print("small")\n')
    assert out == ["small"]


def test_while_loop():
    out = run("i = 0\nwhile i < 3:\n    print(i)\n    i = i + 1\n")
    assert out == ["0", "1", "2"]


def test_for_in_array_break_continue():
    out = run(
        """
        for x in [1, 2, 3, 4, 5]:
            if x == 2:
                continue
            if x == 4:
                break
            print(x)
        """
    )
    assert out == ["1", "3"]


def test_function_call_and_recursion():
    out = run(
        """
        def fact(n: Int) -> Int:
            if n <= 1:
                return 1
            return n * fact(n - 1)
        print(fact(5))
        """
    )
    assert out == ["120"]


def test_closures_capture_environment():
    out = run(
        """
        base = 10
        def add_base(x: Int) -> Int:
            return x + base
        print(add_base(5))
        """
    )
    assert out == ["15"]


def test_arrays_index_and_index_assign():
    out = run(
        """
        xs = [1, 2, 3]
        xs[1] = 99
        print(xs[1])
        print(len(xs))
        """
    )
    assert out == ["99", "3"]


def test_type_annotation_mismatch_on_typed_decl_raises():
    with pytest.raises(SenseTypeError):
        run('x: Int = "nope"\n')


def test_type_annotation_mismatch_on_param_raises():
    with pytest.raises(SenseTypeError):
        run(
            """
            def f(a: Int) -> Int:
                return a
            f("nope")
            """
        )


def test_undefined_variable_raises_name_error():
    with pytest.raises(SenseNameError):
        run("print(nope)\n")


def test_division_by_zero_raises():
    with pytest.raises(SenseRuntimeError):
        run("print(1 / 0)\n")


def test_calling_non_function_raises():
    with pytest.raises(SenseRuntimeError):
        run("x = 5\nx()\n")


def test_builtins_str_int_float_type_of():
    out = run(
        """
        print(str(42))
        print(int("7") + 1)
        print(float(3))
        print(type_of([1,2]))
        print(type_of("s"))
        print(type_of(true))
        """
    )
    assert out == ["42", "8", "3.0", "Array", "String", "Bool"]


def test_push_builtin_mutates_array():
    out = run("xs = [1]\npush(xs, 2)\nprint(xs)\n")
    assert out == ["[1, 2]"]


def test_import_module_and_member_access(tmp_path):
    (tmp_path / "util.sns").write_text(
        "def square(x: Int) -> Int:\n    return x * x\npi = 3\n", encoding="utf-8"
    )
    main = tmp_path / "main.sns"
    main.write_text(
        'import "./util.sns" as util\nprint(util.square(4))\nprint(util.pi)\n', encoding="utf-8"
    )
    output: list[str] = []
    interp = Interpreter(stdout=output.append)
    interp.run_file(str(main))
    assert output == ["16", "3"]


def test_import_missing_module_raises(tmp_path):
    main = tmp_path / "main.sns"
    main.write_text('import "./nope.sns" as m\n', encoding="utf-8")
    with pytest.raises(SenseImportError):
        Interpreter().run_file(str(main))


def test_circular_import_raises(tmp_path):
    (tmp_path / "a.sns").write_text('import "./b.sns" as b\n', encoding="utf-8")
    (tmp_path / "b.sns").write_text('import "./a.sns" as a\n', encoding="utf-8")
    with pytest.raises(SenseImportError):
        Interpreter().run_file(str(tmp_path / "a.sns"))


def test_equality_distinguishes_bool_from_int():
    out = run("print(true == 1)\nprint(1 == 1)\n")
    assert out == ["false", "true"]


def test_logical_short_circuit():
    out = run(
        """
        def boom() -> Bool:
            print("called")
            return true
        if false and boom():
            print("unreachable")
        print("ok")
        """
    )
    assert out == ["ok"]


def test_word_logical_operators_and_or_not():
    out = run(
        """
        print(true and false)
        print(true or false)
        print(not true)
        """
    )
    assert out == ["false", "true", "false"]


def test_assignment_mutates_outer_scope_from_nested_block():
    out = run(
        """
        counter = 0
        i = 0
        while i < 3:
            counter = counter + 1
            i = i + 1
        print(counter)
        """
    )
    assert out == ["3"]


def test_local_shadows_outer_binding_without_mutating_it():
    out = run(
        """
        x = 1
        if true:
            local x = 2
            print(x)
        print(x)
        """
    )
    assert out == ["2", "1"]


def test_local_type_annotation_is_checked():
    with pytest.raises(SenseTypeError):
        run(
            """
            if true:
                local x: Int = "nope"
            """
        )


def test_assignment_inside_block_stays_block_scoped_if_new():
    with pytest.raises(SenseNameError):
        run(
            """
            if true:
                only_inside = 42
            print(only_inside)
            """
        )
