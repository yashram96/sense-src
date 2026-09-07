"""Where a `persistent memory`'s SQLite file lands when no explicit path
is given (`persistent memory name`, no `: "path"`). Previously always
`<name>.sense_memory.db` relative to the process's current working
directory -- meaning running the exact same script from two different
directories silently used two different databases, and every
default-location memory across a whole machine landed in one shared,
unorganized pile. See `Interpreter._default_memory_path`.

An explicit path (covered by tests/test_persistent_memory.py) always wins
and is completely unaffected by anything here.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from sense_lang.interpreter import Interpreter


def write_program(tmp_path: Path, subdir: str = "") -> Path:
    script_dir = tmp_path / subdir if subdir else tmp_path
    script_dir.mkdir(parents=True, exist_ok=True)
    script = script_dir / "prog.sns"
    script.write_text(
        dedent(
            """
            persistent memory ledger
            ledger.remember("k", "v")
            print(ledger.path)
            """
        ),
        encoding="utf-8",
    )
    return script


def test_default_location_is_next_to_the_script_not_the_cwd(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    script = write_program(tmp_path, "project")
    other_cwd = tmp_path / "somewhere_else"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)

    interp = Interpreter(stdout=lambda _s: None)
    env = interp.run_file(str(script))
    mem = env.get("ledger")

    expected = project_dir / ".sense_memory" / "ledger.sense_memory.db"
    assert Path(mem.path).resolve() == expected.resolve()
    assert expected.exists()
    # Nothing landed in the invoking shell's own directory.
    assert not (other_cwd / "ledger.sense_memory.db").exists()
    assert not list(other_cwd.glob("*.db"))


def test_default_location_is_stable_across_different_invoking_directories(tmp_path, monkeypatch):
    """The actual bug: running the *same* script from two different
    terminals/directories used to silently produce two different
    database files. Now it's the same file regardless."""
    script = write_program(tmp_path, "project")

    cwd_a = tmp_path / "cwd_a"
    cwd_b = tmp_path / "cwd_b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    monkeypatch.chdir(cwd_a)
    path_a = Interpreter(stdout=lambda _s: None).run_file(str(script)).get("ledger").path
    monkeypatch.chdir(cwd_b)
    path_b = Interpreter(stdout=lambda _s: None).run_file(str(script)).get("ledger").path

    assert Path(path_a).resolve() == Path(path_b).resolve()


def test_default_location_falls_back_to_cwd_relative_with_no_file_path(tmp_path, monkeypatch):
    """run_source() (the REPL's own path, and most of this test suite's)
    has no real .sns file to anchor to -- the only case where the old
    flat, CWD-relative default still applies, since there's nothing
    better available."""
    monkeypatch.chdir(tmp_path)
    interp = Interpreter(stdout=lambda _s: None)
    env = interp.run_source('persistent memory ledger\nledger.remember("k", "v")\n')
    mem = env.get("ledger")
    assert mem.path == "ledger.sense_memory.db"
    assert (tmp_path / "ledger.sense_memory.db").exists()


def test_sense_memory_dir_env_var_overrides_the_default(tmp_path, monkeypatch):
    script = write_program(tmp_path, "project")
    override_dir = tmp_path / "central_memory"
    monkeypatch.setenv("SENSE_MEMORY_DIR", str(override_dir))

    interp = Interpreter(stdout=lambda _s: None)
    env = interp.run_file(str(script))
    mem = env.get("ledger")

    expected = override_dir / "ledger.sense_memory.db"
    assert Path(mem.path).resolve() == expected.resolve()
    assert expected.exists()
    # The per-file .sense_memory/ subfolder is not used once the env var
    # override is set -- the user explicitly chose one shared location.
    assert not (tmp_path / "project" / ".sense_memory").exists()


def test_explicit_path_still_wins_over_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("SENSE_MEMORY_DIR", str(tmp_path / "should_not_be_used"))
    explicit_path = tmp_path / "exactly_here.db"
    script = tmp_path / "prog.sns"
    script.write_text(
        dedent(
            f'''
            persistent memory ledger: "{str(explicit_path).replace(chr(92), "/")}"
            ledger.remember("k", "v")
            '''
        ),
        encoding="utf-8",
    )
    interp = Interpreter(stdout=lambda _s: None)
    env = interp.run_file(str(script))
    mem = env.get("ledger")
    assert Path(mem.path).resolve() == explicit_path.resolve()
    assert explicit_path.exists()
