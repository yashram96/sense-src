"""`persistent memory` -- durable, versioned, append-only key/value storage
(SQLite-backed, scoped down from literal Delta Lake to the part worth
keeping: an append-only log gives versioning and auditing as the same
mechanism).

Session `memory` (no `persistent` keyword) is untouched by this feature --
see tests/test_memory.py for that.
"""

import sqlite3
import threading
from pathlib import Path
from textwrap import dedent

import pytest

from sense_lang.errors import SenseRuntimeError
from sense_lang.interpreter import Interpreter


def run(source: str, db_path: Path) -> list[str]:
    output: list[str] = []
    source = dedent(source).replace("{db}", str(db_path).replace("\\", "/"))
    Interpreter(stdout=output.append).run_source(source)
    return output


# -- basic remember/recall/forget --------------------------------------------------


def test_remember_and_recall(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.remember("name", "ada")
        print(ledger.recall("name"))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["ada"]


def test_recall_of_unknown_key_is_nil(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        print(ledger.recall("missing"))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["nil"]


def test_remember_overwrites_the_visible_value_but_keeps_history(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.remember("name", "ada")
        ledger.remember("name", "ada lovelace")
        print(ledger.recall("name"))
        print(len(ledger.history("name")))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["ada lovelace", "2"]


def test_forget_is_a_tombstone_not_a_delete(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.remember("name", "ada")
        ledger.forget("name")
        print(ledger.recall("name"))
        print(len(ledger.history("name")))
        """,
        tmp_path / "ledger.db",
    )
    # recall() is nil after forgetting, but history() still shows both events --
    # the whole point of an append-only log over a real delete.
    assert out == ["nil", "2"]


def test_forgetting_an_unknown_key_is_not_an_error(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.forget("nope")
        print("ok")
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["ok"]


def test_keys_reflects_only_live_not_forgotten_keys(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.remember("a", 1)
        ledger.remember("b", 2)
        ledger.forget("a")
        print(ledger.keys())
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["[b]"]


def test_stores_json_serializable_types(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.remember("i", 42)
        ledger.remember("f", 3.5)
        ledger.remember("b", true)
        ledger.remember("n", nil)
        ledger.remember("arr", [1, 2, 3])
        print(ledger.recall("i"))
        print(ledger.recall("f"))
        print(ledger.recall("b"))
        print(ledger.recall("n"))
        print(ledger.recall("arr"))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["42", "3.5", "true", "nil", "[1, 2, 3]"]


# -- versioning / auditing ---------------------------------------------------------


def test_history_lines_are_in_version_order_and_include_both_ops(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.remember("name", "ada")
        ledger.remember("name", "ada lovelace")
        ledger.forget("name")
        for line in ledger.history("name"):
            print(line)
        """,
        tmp_path / "ledger.db",
    )
    assert len(out) == 3
    assert "[v1] remember 'name' = ada" in out[0]
    assert "[v2] remember 'name' = ada lovelace" in out[1]
    assert "[v3] forget 'name'" in out[2]


def test_history_of_a_key_with_no_events_is_empty(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        print(len(ledger.history("never_touched")))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["0"]


def test_as_of_time_travels_to_a_past_version(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.remember("name", "ada")        # version 1
        ledger.remember("name", "ada lovelace") # version 2
        print(ledger.as_of("name", 1))
        print(ledger.as_of("name", 2))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["ada", "ada lovelace"]


def test_as_of_before_the_key_ever_existed_is_nil(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        other = "unrelated"
        ledger.remember("name", "ada")   # version 1
        print(ledger.as_of("name", 0))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["nil"]


def test_as_of_reflects_a_forget_that_happened_by_that_version(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        ledger.remember("name", "ada")   # v1
        ledger.forget("name")             # v2
        ledger.remember("name", "ada again") # v3
        print(ledger.as_of("name", 1))
        print(ledger.as_of("name", 2))
        print(ledger.as_of("name", 3))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["ada", "nil", "ada again"]


# -- errors -------------------------------------------------------------------------


def test_non_json_serializable_value_is_rejected(tmp_path):
    with pytest.raises(SenseRuntimeError):
        run(
            """
            import python "os" as os
            persistent memory ledger: "{db}"
            ledger.remember("mod", os)
            """,
            tmp_path / "ledger.db",
        )


def test_non_string_key_is_rejected(tmp_path):
    with pytest.raises(SenseRuntimeError):
        run(
            """
            persistent memory ledger: "{db}"
            ledger.remember(1, "x")
            """,
            tmp_path / "ledger.db",
        )


def test_as_of_non_int_version_is_rejected(tmp_path):
    with pytest.raises(SenseRuntimeError):
        run(
            """
            persistent memory ledger: "{db}"
            ledger.remember("x", 1)
            ledger.as_of("x", "not an int")
            """,
            tmp_path / "ledger.db",
        )


# -- path handling --------------------------------------------------------------------


def test_default_path_is_derived_from_the_memory_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run("persistent memory notes\n", tmp_path)  # {db} unused here
    assert (tmp_path / "notes.sense_memory.db").exists()


def test_explicit_path_is_used_as_given(tmp_path):
    db = tmp_path / "custom_name.db"
    run('persistent memory ledger: "{db}"\nledger.remember("x", 1)\n', db)
    assert db.exists()


def test_type_of_a_persistent_memory(tmp_path):
    out = run(
        """
        persistent memory ledger: "{db}"
        print(type_of(ledger))
        """,
        tmp_path / "ledger.db",
    )
    assert out == ["Memory"]


def test_path_member_is_readable(tmp_path):
    db = tmp_path / "ledger.db"
    out = run('persistent memory ledger: "{db}"\nprint(ledger.path)\n', db)
    assert out == [str(db).replace("\\", "/")]


# -- the actual persistence claim: durability across interpreter instances ---------


def test_durable_across_separate_interpreter_instances(tmp_path):
    db = tmp_path / "ledger.db"
    run('persistent memory ledger: "{db}"\nledger.remember("name", "ada")\n', db)

    # A brand-new Interpreter, pointed at the same file, sees the earlier write.
    out = run('persistent memory ledger: "{db}"\nprint(ledger.recall("name"))\n', db)
    assert out == ["ada"]


def test_data_survives_in_the_actual_sqlite_file(tmp_path):
    db = tmp_path / "ledger.db"
    run('persistent memory ledger: "{db}"\nledger.remember("x", 42)\n', db)

    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT key, op, value FROM memory_log").fetchall()
    conn.close()
    assert rows == [("x", "remember", "42")]


# -- concurrency: safe under async agent/tool writes -------------------------------


def test_concurrent_async_writes_do_not_corrupt_the_log(tmp_path):
    db = tmp_path / "ledger.db"
    out = run(
        """
        persistent memory ledger: "{db}"

        async agent writer_a:
            i = 0
            while i < 20:
                ledger.remember("counter", i)
                i = i + 1

        async agent writer_b:
            i = 0
            while i < 20:
                ledger.remember("counter", i + 100)
                i = i + 1

        start(writer_a)
        start(writer_b)
        await(writer_a)
        await(writer_b)

        print(len(ledger.history("counter")))
        """,
        db,
    )
    # Every write landed as its own row -- no torn/lost writes under
    # concurrent access from two async agent threads.
    assert out == ["40"]
