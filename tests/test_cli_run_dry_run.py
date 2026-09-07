"""`sense run --dry-run <file>` -- the CLI's banner/summary wrapper around
Interpreter(dry_run=True). The interpreter-level semantics (capability/
approval checks stay real, only the body is skipped) are covered in
tests/test_dry_run.py; this is just the CLI's own output and exit code.
"""

from sense_lang import cli as ini


def test_dry_run_prints_banner_and_suppresses_the_real_effect(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text(
        "reversible action send_mail(to: String) requires email.send:\n"
        '    print("SENDING to " + to)\n'
        "policy:\n"
        "    allow email.send\n"
        'mail = send_mail("alice@example.com")\n'
        "mail.verify()\n"
        "mail.commit()\n",
        encoding="utf-8",
    )
    code = cli.main(["run", "--dry-run", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert "DRY RUN" in out
    assert "SENDING to alice@example.com" not in out
    assert "dry run summary" in out
    assert "committed" in out
    assert "dry run — body not executed" in out


def test_without_the_flag_the_real_effect_still_happens(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text(
        "reversible action send_mail(to: String) requires email.send:\n"
        '    print("SENDING to " + to)\n'
        "policy:\n"
        "    allow email.send\n"
        'mail = send_mail("alice@example.com")\n'
        "mail.verify()\n"
        "mail.commit()\n",
        encoding="utf-8",
    )
    code = cli.main(["run", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert "SENDING to alice@example.com" in out
    assert "DRY RUN" not in out
    assert "dry run summary" not in out


def test_dry_run_with_no_actions_prints_empty_summary(tmp_path, capsys):
    f = tmp_path / "a.sns"
    f.write_text('print("hello")\n', encoding="utf-8")
    code = cli.main(["run", "--dry-run", str(f)])
    out = capsys.readouterr().out
    assert code == 0
    assert "(no actions were prepared)" in out
