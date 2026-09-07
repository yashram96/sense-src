"""`import python "module"` -- reaches straight into any installed
Python package.

Tractable because Sense's own reference interpreter already runs inside
the same Python process: no FFI, no serialization boundary, just
importlib + getattr. What this does and does not gate: capability/policy
only applies to a call wrapped in a Sense `action` -- the same boundary
that already exists for a plain Sense `fn`, not a new hole opened by
interop.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseImportError, SenseRuntimeError, SensePolicyError
from sense_lang.interpreter import Interpreter


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


# -- basic import, call, attribute access ----------------------------------------


def test_import_python_module_and_call_function():
    out = run('import python "math" as math\nprint(math.sqrt(16))\n')
    assert out == ["4.0"]


def test_import_python_module_attribute_access():
    out = run('import python "math" as math\nprint(math.pi)\n')
    assert out == [str(__import__("math").pi)]


def test_default_alias_is_the_last_dotted_segment():
    env = Interpreter(stdout=lambda _: None).run_source('import python "os.path"\nx = path\n')
    assert env.has("path")


def test_type_of_a_python_module():
    out = run('import python "math" as math\nprint(type_of(math))\n')
    assert out == ["Python"]


# -- value conversion: most Sense values ARE Python values already --------------


def test_python_list_becomes_a_sense_array():
    out = run(
        """
        import python "json" as json
        data = json.loads("[1, 2, 3]")
        print(type_of(data))
        print(data)
        """
    )
    assert out == ["Array", "[1, 2, 3]"]


def test_sense_array_passed_into_python_call():
    out = run(
        """
        import python "json" as json
        encoded = json.dumps([1, "two", 3])
        print(encoded)
        """
    )
    assert out == ['[1, "two", 3]']


def test_native_array_builtins_work_on_a_python_returned_list():
    out = run(
        """
        import python "json" as json
        data = json.loads("[1, 2]")
        push(data, 3)
        print(data)
        print(len(data))
        """
    )
    assert out == ["[1, 2, 3]", "3"]


def test_string_int_float_bool_none_pass_through_unwrapped():
    out = run(
        """
        import python "json" as json
        print(type_of(json.loads("42")))
        print(type_of(json.loads("3.5")))
        print(type_of(json.loads("\\"hi\\"")))
        print(type_of(json.loads("true")))
        print(type_of(json.loads("null")))
        """
    )
    assert out == ["Int", "Float", "String", "Bool", "Nil"]


# -- opaque values: dicts and other objects --------------------------------------


def test_dict_is_an_opaque_python_value():
    env = Interpreter(stdout=lambda _: None).run_source(
        'import python "json" as json\nobj = json.loads("{}")\n'
    )
    assert env.get("obj").__class__.__name__ == "PythonValue"


def test_dict_supports_index_read_and_write():
    out = run(
        """
        import python "json" as json
        obj = json.loads("{\\"name\\": \\"ada\\", \\"age\\": 36}")
        print(obj["name"])
        obj["age"] = 37
        print(obj["age"])
        """
    )
    assert out == ["ada", "37"]


def test_member_access_on_a_wrapped_object():
    out = run(
        """
        import python "datetime" as datetime
        d = datetime.date(2020, 1, 15)
        print(d.year)
        print(d.month)
        print(d.day)
        """
    )
    assert out == ["2020", "1", "15"]


def test_calling_a_method_on_a_wrapped_object():
    out = run(
        """
        import python "datetime" as datetime
        d = datetime.date(2020, 1, 15)
        print(d.isoformat())
        """
    )
    assert out == ["2020-01-15"]


def test_printing_a_wrapped_object_shows_its_own_str_not_a_placeholder():
    # print() on a PythonValue used to show a useless "<python date>"
    # placeholder instead of the wrapped object's real string form.
    out = run(
        """
        import python "datetime" as datetime
        d = datetime.date(2020, 1, 15)
        print(d)
        """
    )
    assert out == ["2020-01-15"]


# -- error handling: never a raw Python traceback --------------------------------


def test_missing_module_raises_sense_import_error():
    with pytest.raises(SenseImportError):
        run('import python "this_module_does_not_exist_xyz" as m\n')


def test_missing_attribute_raises_sense_runtime_error():
    with pytest.raises(SenseRuntimeError):
        run('import python "math" as math\nprint(math.not_a_real_thing)\n')


def test_python_call_error_is_wrapped():
    with pytest.raises(SenseRuntimeError):
        run('import python "math" as math\nprint(math.sqrt("nope"))\n')


def test_python_call_error_message_names_the_exception_type():
    # A bare exception message can be meaningless out of context (e.g. a
    # KeyError's str() is just the key). Naming the exception type keeps
    # the error legible.
    with pytest.raises(SenseRuntimeError, match="TypeError"):
        run('import python "math" as math\nprint(math.sqrt("nope"))\n')


def test_python_index_error_message_names_the_exception_type():
    with pytest.raises(SenseRuntimeError, match="KeyError"):
        run(
            """
            import python "json" as json
            obj = json.loads("{}")
            print(obj["missing"])
            """
        )


# -- the safety story: gating only applies inside an action ----------------------


def test_bare_python_call_is_not_policy_gated():
    # No `action` wrapping -- behaves like calling any plain Sense fn: it
    # just runs. This mirrors the existing, already-documented boundary
    # (an ordinary fn's body isn't capability-checked either), not a new
    # hole introduced by interop.
    out = run(
        """
        import python "math" as math
        policy:
            deny anything.at.all
        print(math.sqrt(9))
        """
    )
    assert out == ["3.0"]


def test_python_call_wrapped_in_action_is_policy_gated():
    with pytest.raises(SensePolicyError):
        run(
            """
            import python "math" as math

            irreversible action compute(x: Int) requires math.compute:
                math.sqrt(x)

            policy:
                deny math.compute

            a = compute(9)
            a.verify()
            """
        )


def test_python_call_wrapped_in_action_succeeds_when_allowed():
    out = run(
        """
        import python "math" as math

        irreversible action compute(x: Int) requires math.compute:
            return math.sqrt(x)

        policy:
            allow math.compute

        a = compute(16)
        a.verify()
        print(a.commit())
        """
    )
    assert out == ["4.0"]
