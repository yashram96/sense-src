"""`memory` — a persistent-for-the-run key/value store, distinct from an
ordinary variable.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseRuntimeError
from sense_lang.interpreter import Interpreter


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


def test_remember_and_recall():
    out = run(
        """
        memory notes
        notes.remember("topic", "sense keywords")
        print(notes.recall("topic"))
        """
    )
    assert out == ["sense keywords"]


def test_recall_of_unknown_key_is_nil_not_an_error():
    out = run(
        """
        memory notes
        print(notes.recall("missing"))
        """
    )
    assert out == ["nil"]


def test_forget_removes_a_key():
    out = run(
        """
        memory notes
        notes.remember("a", 1)
        notes.forget("a")
        print(notes.recall("a"))
        print(notes.keys())
        """
    )
    assert out == ["nil", "[]"]


def test_forgetting_an_unknown_key_is_not_an_error():
    out = run(
        """
        memory notes
        notes.forget("nope")
        print("ok")
        """
    )
    assert out == ["ok"]


def test_keys_lists_all_remembered_keys():
    out = run(
        """
        memory notes
        notes.remember("a", 1)
        notes.remember("b", 2)
        print(len(notes.keys()))
        """
    )
    assert out == ["2"]


def test_kind_is_a_free_form_optional_tag():
    out = run(
        """
        memory short_term: "short-term"
        memory untagged
        print(short_term.kind)
        print(untagged.kind)
        """
    )
    assert out == ["short-term", "nil"]


def test_kind_is_not_validated_against_a_fixed_set():
    # Categories are "potential," not a closed enum.
    out = run(
        """
        memory m: "whatever-i-want"
        print(m.kind)
        """
    )
    assert out == ["whatever-i-want"]


def test_memory_key_must_be_a_string():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            memory notes
            notes.remember(1, "x")
            """
        )


def test_type_of_a_memory():
    out = run(
        """
        memory notes
        print(type_of(notes))
        """
    )
    assert out == ["Memory"]


def test_memory_is_a_scoped_binding_like_model():
    out = run(
        """
        session s:
            memory notes
            notes.remember("x", 1)
            print(notes.recall("x"))
        """
    )
    assert out == ["1"]
