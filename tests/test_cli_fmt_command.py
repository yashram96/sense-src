"""`sense fmt <file|dir>` -- default (stdout), --write, --check, and error
handling. The formatting logic itself is covered in tests/test_formatter.py;
this is just the CLI's file discovery, mode dispatch, and exit codes."""

from sense_lang import cli


def test_default_mode_prints_formatted_source_to_stdout(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text("x = 1 + 2 * 3\n", encoding="utf-8")
    code = cli.main(["fmt", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert out == "x = 1 + 2 * 3\n"


def test_default_mode_does_not_write_the_file(tmp_path):
    f = tmp_path / "a.sns"
    original = "policy:\n    allow x.y\n"
    f.write_text(original, encoding="utf-8")
    cli.main(["fmt", str(f)])
    assert f.read_text(encoding="utf-8") == original


def test_write_mode_rewrites_a_changed_file(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text("policy:\n    allow x.y\n", encoding="utf-8")
    code = cli.main(["fmt", str(f), "--write"])
    out = capsys.readouterr().out
    assert code == 0
    assert f.read_text(encoding="utf-8") == "policy: allow x.y\n"
    assert str(f) in out


def test_write_mode_leaves_an_already_formatted_file_untouched(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text("x = 1\n", encoding="utf-8")
    mtime_before = f.stat().st_mtime_ns
    code = cli.main(["fmt", str(f), "--write"])
    out = capsys.readouterr().out
    assert code == 0
    assert out == ""  # nothing changed, nothing printed
    assert f.stat().st_mtime_ns == mtime_before


def test_check_mode_reports_unformatted_file_and_exits_one(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text("policy:\n    allow x.y\n", encoding="utf-8")
    code = cli.main(["fmt", str(f), "--check"])
    out = capsys.readouterr().out
    assert code == 1
    assert str(f) in out
    # --check never writes
    assert f.read_text(encoding="utf-8") == "policy:\n    allow x.y\n"


def test_check_mode_on_already_formatted_file_exits_zero(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text("x = 1\n", encoding="utf-8")
    code = cli.main(["fmt", str(f), "--check"])
    out = capsys.readouterr().out
    assert code == 0
    assert out == ""


def test_directory_mode_formats_every_sns_file(tmp_path, capsys):
    (tmp_path / "a.sns").write_text("policy:\n    allow x.y\n", encoding="utf-8")
    (tmp_path / "b.sns").write_text("x = 1\n", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "c.sns").write_text("policy:\n    deny p.q\n", encoding="utf-8")

    code = cli.main(["fmt", str(tmp_path), "--check"])
    out = capsys.readouterr().out
    assert code == 1  # a.sns and c.sns need reformatting
    assert "a.sns" in out
    assert "c.sns" in out
    assert "b.sns" not in out  # already formatted


def test_directory_mode_ignores_non_sns_files(tmp_path, capsys):
    (tmp_path / "readme.txt").write_text("not sense code", encoding="utf-8")
    code = cli.main(["fmt", str(tmp_path), "--check"])
    out = capsys.readouterr().out
    assert code == 0
    assert "no .sns files found" in out


def test_missing_path_is_an_error(tmp_path, capsys):
    code = cli.main(["fmt", str(tmp_path / "nope.sns")])
    err = capsys.readouterr().err
    assert code == 1
    assert "no such file or directory" in err


def test_syntax_error_is_reported_and_does_not_crash(tmp_path, capsys):
    f = tmp_path / "broken.sns"
    f.write_text("x = 1 +\n", encoding="utf-8")
    code = cli.main(["fmt", str(f)])
    err = capsys.readouterr().err
    assert code == 1
    assert "SenseError" in err


def test_multiple_files_get_a_header_in_stdout_mode(tmp_path, capsys):
    (tmp_path / "a.sns").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "b.sns").write_text("y = 2\n", encoding="utf-8")
    code = cli.main(["fmt", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "a.sns" in out and "b.sns" in out
    assert "x = 1" in out and "y = 2" in out
