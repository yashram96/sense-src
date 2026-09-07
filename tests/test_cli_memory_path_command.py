"""`sense memory-path [file]` -- shows where a file's `persistent memory`
declarations would store their SQLite files, without running the file at
all (parses it -- lexer + parser only, no execution, since a real
`persistent memory` declaration has a side effect -- creating the file --
the moment it actually runs). See Interpreter._default_memory_path.
"""

from textwrap import dedent

from sense_lang import cli


def test_no_file_reports_no_override_set(monkeypatch, capsys):
    monkeypatch.delenv("SENSE_MEMORY_DIR", raising=False)
    code = cli.main(["memory-path"])
    out = capsys.readouterr().out
    assert code == 0
    assert "SENSE_MEMORY_DIR is not set" in out


def test_no_file_reports_the_override_when_set(monkeypatch, capsys):
    monkeypatch.setenv("SENSE_MEMORY_DIR", "/wherever/i/chose")
    code = cli.main(["memory-path"])
    out = capsys.readouterr().out
    assert code == 0
    assert "/wherever/i/chose" in out


def test_a_file_with_no_persistent_memory_says_so(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text("x = 1\n", encoding="utf-8")
    code = cli.main(["memory-path", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert "declares no 'persistent memory'" in out


def test_default_location_is_reported_without_running_the_file(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SENSE_MEMORY_DIR", raising=False)
    f = tmp_path / "a.sns"
    f.write_text("persistent memory ledger\n", encoding="utf-8")
    code = cli.main(["memory-path", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert "ledger:" in out
    assert str(tmp_path / ".sense_memory" / "ledger.sense_memory.db") in out
    assert "default" in out
    # The whole point: this must not have actually created the file.
    assert not (tmp_path / ".sense_memory").exists()


def test_sense_memory_dir_override_is_reflected(tmp_path, capsys, monkeypatch):
    override = tmp_path / "central"
    monkeypatch.setenv("SENSE_MEMORY_DIR", str(override))
    f = tmp_path / "a.sns"
    f.write_text("persistent memory ledger\n", encoding="utf-8")
    code = cli.main(["memory-path", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert str(override / "ledger.sense_memory.db") in out
    assert "SENSE_MEMORY_DIR" in out


def test_explicit_path_is_reported_as_explicit_not_default(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text('persistent memory ledger: "somewhere.db"\n', encoding="utf-8")
    code = cli.main(["memory-path", str(f)])
    out = capsys.readouterr().out
    assert "ledger: somewhere.db" in out
    assert "explicit path" in out


def test_finds_declarations_nested_inside_session_and_agent_bodies(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("SENSE_MEMORY_DIR", raising=False)
    f = tmp_path / "a.sns"
    f.write_text(
        dedent(
            """
            session work:
                persistent memory nested_ledger

            agent worker:
                persistent memory agent_ledger
            """
        ),
        encoding="utf-8",
    )
    code = cli.main(["memory-path", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert "nested_ledger:" in out
    assert "agent_ledger:" in out


def test_session_memory_not_persistent_is_not_listed(tmp_path, capsys):
    """A plain (session) `memory` has no file at all -- only `persistent
    memory` is relevant here."""
    f = tmp_path / "a.sns"
    f.write_text("memory notes\n", encoding="utf-8")
    code = cli.main(["memory-path", str(f)])
    out = capsys.readouterr().out
    assert "declares no 'persistent memory'" in out


def test_a_syntax_error_is_reported_not_a_traceback(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text("this is not valid sense (((\n", encoding="utf-8")
    code = cli.main(["memory-path", str(f)])
    err = capsys.readouterr().err
    assert code == 1
    assert "SenseError" in err


def test_a_missing_file_is_reported_not_a_traceback(tmp_path, capsys):
    code = cli.main(["memory-path", str(tmp_path / "does_not_exist.sns")])
    err = capsys.readouterr().err
    assert code == 1
    assert "SenseError" in err
