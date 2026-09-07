"""`sense test <file|dir>` — CLI discovery and reporting on top of the
`test` statement covered in tests/test_test_statement.py."""

from sense_lang import cli


def test_single_file_all_passing_exits_zero(tmp_path, capsys):
    f = tmp_path / "anything.sns"
    f.write_text('test "ok":\n    assert(1 == 1)\n', encoding="utf-8")
    code = cli.main(["test", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS  ok" in out
    assert "1 file, 1 test: 1 passed, 0 failed" in out


def test_single_file_with_failure_exits_one(tmp_path, capsys):
    f = tmp_path / "anything.sns"
    f.write_text('test "bad":\n    assert(1 == 2, "nope")\n', encoding="utf-8")
    code = cli.main(["test", str(f)])
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL  bad" in out
    assert "nope" in out
    assert "1 file, 1 test: 0 passed, 1 failed" in out


def test_a_file_with_no_test_blocks_reports_zero_tests(tmp_path, capsys):
    f = tmp_path / "anything.sns"
    f.write_text('print("hi")\n', encoding="utf-8")
    code = cli.main(["test", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 file, 0 tests: 0 passed, 0 failed" in out


def test_directory_discovers_test_prefixed_and_suffixed_files_only(tmp_path, capsys):
    (tmp_path / "test_a.sns").write_text('test "a":\n    assert(true)\n', encoding="utf-8")
    (tmp_path / "b_test.sns").write_text('test "b":\n    assert(true)\n', encoding="utf-8")
    (tmp_path / "not_discovered.sns").write_text('test "c":\n    assert(false)\n', encoding="utf-8")
    code = cli.main(["test", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0  # the failing, non-matching file must NOT have been picked up
    assert "2 files, 2 tests: 2 passed, 0 failed" in out


def test_directory_discovers_nested_test_files(tmp_path, capsys):
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "test_nested.sns").write_text('test "n":\n    assert(true)\n', encoding="utf-8")
    code = cli.main(["test", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "1 file, 1 test: 1 passed, 0 failed" in out


def test_empty_directory_reports_no_test_files_found(tmp_path, capsys):
    code = cli.main(["test", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "no test files found" in out


def test_missing_path_is_an_error(tmp_path, capsys):
    code = cli.main(["test", str(tmp_path / "nope.sns")])
    err = capsys.readouterr().err
    assert code == 1
    assert "no such file or directory" in err


def test_syntax_error_outside_any_test_block_is_a_file_level_error(tmp_path, capsys):
    f = tmp_path / "broken.sns"
    f.write_text('test "will not parse":\n    x = 1 +\n', encoding="utf-8")
    code = cli.main(["test", str(f)])
    captured = capsys.readouterr()
    assert code == 1
    assert "ERROR" in captured.err
    assert "0 tests" in captured.out


def test_multiple_files_aggregate_totals(tmp_path, capsys):
    (tmp_path / "test_one.sns").write_text(
        'test "a":\n    assert(true)\ntest "b":\n    assert(false)\n', encoding="utf-8"
    )
    (tmp_path / "test_two.sns").write_text('test "c":\n    assert(true)\n', encoding="utf-8")
    code = cli.main(["test", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "2 files, 3 tests: 2 passed, 1 failed" in out
