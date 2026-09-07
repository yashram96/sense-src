"""`sense` with no subcommand: drops into the REPL, matching the same
convenience bare `python` has -- `sense --help`/`-h` (argparse's own,
unaffected by this) is still how the subcommands get discovered."""

from sense_lang import cli


def test_bare_invocation_launches_repl(monkeypatch, capsys):
    inputs = iter(['print("hi from bare sense")', "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    code = cli.main([])

    out = capsys.readouterr().out
    assert code == 0
    assert "REPL" in out  # startup banner
    assert "hi from bare sense" in out


def test_repl_subcommand_still_works_explicitly(monkeypatch, capsys):
    inputs = iter(["exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    code = cli.main(["repl"])

    assert code == 0
    assert "REPL" in capsys.readouterr().out
