"""`sense` command-line entry point: run scripts, start a REPL, run tests."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

from . import ast_nodes as n
from . import __version__
from .errors import SenseError
from .formatter import format_source
from .inspector import run_inspector
from .interpreter import Interpreter
from .lexer import tokenize
from .parser import parse


def _run(path: str, dry_run: bool = False) -> int:
    interpreter = Interpreter(dry_run=dry_run)
    if dry_run:
        print("DRY RUN — no actions will actually commit\n")
    try:
        interpreter.run_file(path)
    except SenseError as exc:
        print(f"SenseError: {exc}", file=sys.stderr)
        return 1
    except RecursionError:
        print("SenseError: maximum recursion depth exceeded", file=sys.stderr)
        return 1
    if dry_run:
        print("\n--- dry run summary ---")
        if interpreter.audit_log:
            for entry in interpreter.audit_log:
                print(entry.describe())
        else:
            print("(no actions were prepared)")
    return 0


def _discover_test_files(path: str) -> list[str]:
    """A single file is always a test target regardless of its name; a
    directory is searched recursively for the pytest-style `test_*.sns` /
    `*_test.sns` naming convention."""
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        matches = sorted(set(p.rglob("test_*.sns")) | set(p.rglob("*_test.sns")))
        return [str(m) for m in matches]
    raise FileNotFoundError(path)


def _test(path: str) -> int:
    """Runs every `test "..." : ...` block found while executing each
    discovered file. A `test` block's own failure is isolated (see
    Interpreter._exec_TestStmt) and doesn't stop the rest of the file or
    the rest of the run; a failure *outside* any test block (a syntax
    error, or an unhandled error in ordinary top-level code) is reported
    as a file-level error instead, since it isn't a named test result."""
    try:
        files = _discover_test_files(path)
    except FileNotFoundError:
        print(f"SenseError: no such file or directory: {path}", file=sys.stderr)
        return 1
    if not files:
        print(f"no test files found under {path} (looked for test_*.sns / *_test.sns)")
        return 0

    total = passed = failed = 0
    any_file_error = False

    for file in files:
        print(file)
        interpreter = Interpreter()
        try:
            interpreter.run_file(file)
        except SenseError as exc:
            print(f"  ERROR  {exc}", file=sys.stderr)
            any_file_error = True
        except RecursionError:
            print("  ERROR  maximum recursion depth exceeded", file=sys.stderr)
            any_file_error = True

        for result in interpreter.test_results:
            total += 1
            if result.passed:
                passed += 1
                print(f"  PASS  {result.description}")
            else:
                failed += 1
                print(f"  FAIL  {result.description}")
                print(f"        {result.error}")
        print()

    file_word = "file" if len(files) == 1 else "files"
    test_word = "test" if total == 1 else "tests"
    print(f"{len(files)} {file_word}, {total} {test_word}: {passed} passed, {failed} failed")
    return 1 if (failed > 0 or any_file_error) else 0


def _discover_sns_files(path: str) -> list[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(str(m) for m in p.rglob("*.sns"))
    raise FileNotFoundError(path)


def _fmt(path: str, write: bool, check: bool) -> int:
    """`sense fmt`: reformat one file or every `.sns` file under a directory.

    Default (no flags) prints the formatted source to stdout — safe and
    non-destructive, since `sense fmt` doesn't preserve every original
    blank line and shifts embedded source line numbers, so overwriting
    deserves an explicit opt-in. `--write` overwrites changed files in place;
    `--check` reports which files would change without writing anything,
    exiting 1 if any would (useful in CI)."""
    try:
        files = _discover_sns_files(path)
    except FileNotFoundError:
        print(f"SenseError: no such file or directory: {path}", file=sys.stderr)
        return 1
    if not files:
        print(f"no .sns files found under {path}")
        return 0

    # format_source() always produces '\n'-terminated text, matching every
    # .sns file already in this codebase (git-diff noise is real cost) --
    # reconfigure stdout so plain print() doesn't translate it to '\r\n' on
    # Windows. `open(..., newline="\n")` gets the same treatment for writes.
    try:
        sys.stdout.reconfigure(newline="\n")
    except (AttributeError, ValueError):
        pass  # not a reconfigurable text stream (e.g. redirected in a way that doesn't support it) -- best effort

    any_changed = False
    any_error = False
    multiple = len(files) > 1

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                original = f.read()
            formatted = format_source(original)
        except SenseError as exc:
            print(f"{file}: SenseError: {exc}", file=sys.stderr)
            any_error = True
            continue

        changed = formatted != original
        any_changed = any_changed or changed

        if check:
            if changed:
                print(file)
        elif write:
            if changed:
                with open(file, "w", encoding="utf-8", newline="\n") as f:
                    f.write(formatted)
                print(file)
        else:
            if multiple:
                print(f"# ----- {file} -----")
            print(formatted, end="")

    if any_error:
        return 1
    return 1 if (check and any_changed) else 0


def _find_persistent_memory_decls(node: object) -> list[n.MemoryDecl]:
    """Every `persistent memory` declaration reachable anywhere in `node`
    (a parsed `Program`), regardless of nesting depth -- inside a
    `session`/`agent`/`if`/`for`/... body, not just at the top level.
    Walks generically over every dataclass field rather than special-
    casing each statement type's own "where's the nested block" shape
    (`Block.statements`, `ActionDecl.rollback_body`, `IfStmt.else_branch`,
    ...) -- future-proof against a new statement type without needing an
    update here specifically. Used by `sense memory-path`, which
    deliberately never runs the file (see `Interpreter._default_memory_path`'s
    own docstring for why: a `persistent memory` declaration can have a
    real side effect -- creating the file -- the moment it actually
    executes, which a pure "where would this go" query shouldn't trigger)."""
    found: list[n.MemoryDecl] = []

    def visit(obj: object) -> None:
        if isinstance(obj, n.MemoryDecl):
            if obj.is_persistent:
                found.append(obj)
            return
        if isinstance(obj, list):
            for item in obj:
                visit(item)
            return
        if is_dataclass(obj):
            for f in fields(obj):
                visit(getattr(obj, f.name))

    visit(node)
    return found


def _memory_path(path: str | None) -> int:
    """`sense memory-path [file]`: where would each `persistent memory`
    declared in `file` actually put its SQLite file, without running the
    file at all (parses it -- lexer + parser only). With no file, just
    reports whether `SENSE_MEMORY_DIR` is currently set."""
    override_dir = os.environ.get("SENSE_MEMORY_DIR")

    if path is None:
        if override_dir:
            print(f"SENSE_MEMORY_DIR is set: {override_dir}")
            print("Every persistent memory declared with no explicit path resolves")
            print(f"flat under this directory: {Path(override_dir) / '<name>.sense_memory.db'}")
        else:
            print("SENSE_MEMORY_DIR is not set.")
            print("A persistent memory declared with no explicit path resolves to a")
            print("'.sense_memory/' folder next to whichever .sns file declares it:")
            print("  <that file's own directory>/.sense_memory/<name>.sense_memory.db")
        print()
        print("Pass a file to see the exact resolved path(s) for what it declares:")
        print("  sense memory-path <file.sns>")
        print()
        print("To change the default, set SENSE_MEMORY_DIR (PowerShell):")
        print('  $env:SENSE_MEMORY_DIR = "C:\\path\\to\\wherever"          # this session only')
        print('  [Environment]::SetEnvironmentVariable("SENSE_MEMORY_DIR", "C:\\path\\to\\wherever", "User")')
        print("                                                          # persists across sessions")
        return 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        program = parse(tokenize(source))
    except SenseError as exc:
        print(f"SenseError: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"SenseError: {exc}", file=sys.stderr)
        return 1

    decls = _find_persistent_memory_decls(program)
    if not decls:
        print(f"{path} declares no 'persistent memory'.")
        return 0

    for decl in decls:
        if decl.kind is not None:
            resolved = decl.kind
            note = "explicit path"
        else:
            resolved = Interpreter._default_memory_path(decl.name, path)
            note = "default -- via SENSE_MEMORY_DIR" if override_dir else "default -- next to this file"
        print(f"{decl.name}: {resolved}  ({note})")
    return 0


def _run_buffer(interpreter: Interpreter, env, buffer: str) -> None:
    try:
        tokens = tokenize(buffer)
        program = parse(tokens)
        for stmt in program.statements:
            interpreter._execute(stmt, env, None)  # noqa: SLF001 (REPL is intentionally low-level)
    except SenseError as exc:
        print(f"SenseError: {exc}", file=sys.stderr)


def _repl() -> int:
    # Sense blocks are indentation, so a multi-line entry (a function/if/
    # while/for header) stays open until a blank line closes it — same
    # convention as Python's own REPL.
    print(f"Sense {__version__} — interactive REPL ('exit' to quit)")
    interpreter = Interpreter()
    env = interpreter.globals.child()
    buffer = ""
    awaiting_block = False
    while True:
        try:
            line = input("...    " if awaiting_block else "sense> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not buffer and not awaiting_block and line.strip() in ("exit", "quit"):
            return 0

        stripped = line.rstrip()
        if awaiting_block:
            if stripped.strip() == "":
                _run_buffer(interpreter, env, buffer)
                buffer = ""
                awaiting_block = False
            else:
                buffer += line + "\n"
            continue

        buffer += line + "\n"
        if stripped.endswith(":"):
            awaiting_block = True
            continue
        _run_buffer(interpreter, env, buffer)
        buffer = ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sense", description="Sense language CLI")
    parser.add_argument("--version", action="version", version=f"sense {__version__}")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run a .sns file")
    run_p.add_argument("path")
    run_p.add_argument(
        "--dry-run", action="store_true",
        help="run normally, but every action's .commit()/.rollback() is simulated, never executing its body",
    )

    sub.add_parser("repl", help="start an interactive REPL")

    test_p = sub.add_parser("test", help="run test blocks in a .sns file or directory")
    test_p.add_argument("path")

    fmt_p = sub.add_parser("fmt", help="format a .sns file or directory")
    fmt_p.add_argument("path")
    fmt_p.add_argument("--write", "-w", action="store_true", help="overwrite files in place")
    fmt_p.add_argument(
        "--check", action="store_true", help="exit 1 if any file isn't already formatted; writes nothing"
    )

    inspect_p = sub.add_parser(
        "inspect", help="run a .sns file and serve a live console over its declared tools/actions/agents/memory"
    )
    inspect_p.add_argument("path")
    inspect_p.add_argument("--port", type=int, default=4300, help="port to serve the console on (default: 4300)")
    inspect_p.add_argument(
        "--dry-run", action="store_true",
        help="every action's .commit()/.rollback() clicked in the console is simulated, never executing its body",
    )

    memory_path_p = sub.add_parser(
        "memory-path",
        help="show where a file's 'persistent memory' declarations would store their SQLite files, without running the file",
    )
    memory_path_p.add_argument(
        "path", nargs="?", default=None,
        help="a .sns file to check (omit to just report whether SENSE_MEMORY_DIR is set)",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        return _run(args.path, args.dry_run)
    if args.command == "repl":
        return _repl()
    if args.command == "test":
        return _test(args.path)
    if args.command == "fmt":
        return _fmt(args.path, args.write, args.check)
    if args.command == "inspect":
        return run_inspector(args.path, args.port, args.dry_run)
    if args.command == "memory-path":
        return _memory_path(args.path)

    # No subcommand given: same convenience `python` itself has -- drop
    # into the REPL rather than just printing help. `sense --help`/`-h`
    # (argparse's own, handled before we get here) is still how you
    # discover the subcommands.
    return _repl()


if __name__ == "__main__":
    raise SystemExit(main())
