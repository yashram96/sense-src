"""`dry_run`: every action's .commit()/.rollback() is simulated rather than
executed. Capability/approval checks stay completely real (a dry run must
tell you honestly whether policy would have blocked the effect) -- only
the body that would perform the real effect is skipped. Deliberately does
not touch `tool` calls -- a `tool` has no prepare/verify/commit staging
to intercept in the first place.
"""

from __future__ import annotations

from textwrap import dedent

import pytest

from sense_lang.errors import SenseApprovalError, SensePolicyError
from sense_lang.interpreter import Interpreter


def run(source: str, dry_run: bool = False) -> tuple[Interpreter, list[str]]:
    output: list[str] = []
    interpreter = Interpreter(stdout=output.append, dry_run=dry_run)
    interpreter.run_source(dedent(source))
    return interpreter, output


def test_dry_run_commit_does_not_run_the_body():
    interp, out = run(
        """
        irreversible action send_mail(to: String) requires email.send:
            print("SENDING to " + to)
        policy:
            allow email.send
        mail = send_mail("alice@example.com")
        mail.verify()
        mail.commit()
        print("state: " + mail.state)
        """,
        dry_run=True,
    )
    assert "SENDING to alice@example.com" not in out
    assert out == ["state: committed"]


def test_dry_run_commit_still_transitions_to_committed():
    interp, out = run(
        """
        irreversible action send_mail(to: String) requires email.send:
            print("SENDING to " + to)
        policy:
            allow email.send
        mail = send_mail("alice@example.com")
        mail.verify()
        mail.commit()
        """,
        dry_run=True,
    )
    assert any("committed" in e.describe() for e in interp.audit_log)


def test_dry_run_commit_audit_entry_is_tagged():
    interp, _out = run(
        """
        irreversible action send_mail(to: String) requires email.send:
            print("SENDING to " + to)
        policy:
            allow email.send
        mail = send_mail("alice@example.com")
        mail.verify()
        mail.commit()
        """,
        dry_run=True,
    )
    committed = [e for e in interp.audit_log if e.event == "committed"]
    assert len(committed) == 1
    assert committed[0].detail == "dry run — body not executed"


def test_dry_run_still_enforces_denied_capability_on_verify():
    """`.verify()` is the first checkpoint a denied capability hits --
    unaffected by dry_run either way, since it never touched the effect."""
    with pytest.raises(SensePolicyError):
        run(
            """
            irreversible action delete_account(user: String) requires account.delete:
                print("DELETING " + user)
            policy:
                deny account.delete
            a = delete_account("bob")
            a.verify()
            """,
            dry_run=True,
        )


def test_dry_run_still_enforces_denied_capability_on_commit():
    """Policy changes between .verify() and .commit() -- .commit()'s own
    independent re-check still fires in dry-run mode."""
    with pytest.raises(SensePolicyError):
        run(
            """
            irreversible action delete_account(user: String) requires account.delete:
                print("DELETING " + user)
            policy:
                allow account.delete
            a = delete_account("bob")
            a.verify()
            policy:
                deny account.delete
            a.commit()
            """,
            dry_run=True,
        )


def test_dry_run_still_enforces_missing_approval_on_commit():
    with pytest.raises(SenseApprovalError):
        run(
            """
            irreversible action delete_account(user: String) requires account.delete, approval:
                print("DELETING " + user)
            policy:
                allow account.delete
            a = delete_account("bob")
            a.verify()
            a.commit()
            """,
            dry_run=True,
        )


def test_dry_run_rollback_does_not_run_rollback_body():
    interp, out = run(
        """
        reversible action update_profile(user: String) requires profile.write:
            print("SETTING " + user)
        rollback:
            print("REVERTING " + user)
        policy:
            allow profile.write
        a = update_profile("alice")
        a.verify()
        a.commit()
        a.rollback()
        print("state: " + a.state)
        """,
        dry_run=True,
    )
    assert "REVERTING alice" not in out
    # The commit itself is also simulated in dry-run mode, so neither
    # body's print ever fires.
    assert "SETTING alice" not in out
    assert out == ["state: rolled_back"]


def test_dry_run_rollback_still_enforces_denied_capability():
    # Deny the capability only after commit, then attempt rollback --
    # mirrors examples/action_rollback.sns's own "policy changes between
    # commit and rollback" scenario.
    with pytest.raises(SensePolicyError):
        run(
            """
            reversible action update_profile(user: String) requires profile.write:
                print("SETTING " + user)
            rollback:
                print("REVERTING " + user)
            policy:
                allow profile.write
            a = update_profile("alice")
            a.verify()
            a.commit()
            policy:
                deny profile.write
            a.rollback()
            """,
            dry_run=True,
        )


def test_normal_run_is_completely_unaffected_by_dry_run_flag_default():
    """Regression guard: dry_run defaults to False, and a normal run's
    commit still actually executes its body."""
    interp, out = run(
        """
        irreversible action send_mail(to: String) requires email.send:
            print("SENDING to " + to)
        policy:
            allow email.send
        mail = send_mail("alice@example.com")
        mail.verify()
        mail.commit()
        """
    )
    assert out == ["SENDING to alice@example.com"]
    committed = [e for e in interp.audit_log if e.event == "committed"]
    assert committed[0].detail is None


def test_tool_calls_are_unaffected_by_dry_run():
    """`tool` has no prepare/verify/commit staging to intercept -- dry-run
    scope is deliberately actions only."""
    interp, out = run(
        """
        tool search(query: String) returns String:
            print("searching for " + query)
            return "results for " + query
        print(search("sense"))
        """,
        dry_run=True,
    )
    assert out == ["searching for sense", "results for sense"]
