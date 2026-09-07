"""Phase 4 follow-ups: a declared 'requires approval' marker on `action`,
and an audit log recording every prepared/verified/committed/denied/failed
event. Both close gaps explicitly tracked in docs/ROADMAP.md after
ask_human shipped."""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseApprovalError, SensePolicyError, SenseRuntimeError
from sense_lang.interpreter import Interpreter


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


def run_interp(source: str) -> Interpreter:
    interp = Interpreter(stdout=lambda _: None)
    interp.run_source(dedent(source))
    return interp


# -- requires approval ------------------------------------------------------------


def test_commit_without_approval_raises_approval_error():
    with pytest.raises(SenseApprovalError):
        run(
            """
            irreversible action f() requires approval:
                print("ran")
            a = f()
            a.verify()
            a.commit()
            """
        )


def test_approve_then_commit_succeeds():
    out = run(
        """
        irreversible action f() requires approval:
            print("ran")
        a = f()
        a.verify()
        a.approve()
        a.commit()
        """
    )
    assert out == ["ran"]


def test_approved_defaults_to_false():
    env = Interpreter(stdout=lambda _: None).run_source(
        dedent(
            """
            irreversible action f() requires approval:
                print(1)
            a = f()
            """
        )
    )
    assert env.get("a").approved is False


def test_approve_sets_approved_true():
    env = Interpreter(stdout=lambda _: None).run_source(
        dedent(
            """
            irreversible action f() requires approval:
                print(1)
            a = f()
            a.approve()
            """
        )
    )
    assert env.get("a").approved is True


def test_requires_capability_and_approval_together():
    with pytest.raises(SensePolicyError):
        run(
            """
            irreversible action f() requires x.y, approval:
                print("ran")
            policy:
                deny x.y
            a = f()
            a.verify()
            """
        )


def test_requires_capability_and_approval_in_reverse_order():
    out = run(
        """
        irreversible action f() requires approval, x.y:
            print("ran")
        policy:
            allow x.y
        a = f()
        a.verify()
        a.approve()
        a.commit()
        """
    )
    assert out == ["ran"]


def test_approval_alone_ignores_capability_policy():
    # no `requires x.y` at all -- policy is irrelevant, only approval gates it
    out = run(
        """
        irreversible action f() requires approval:
            print("ran")
        a = f()
        a.verify()
        a.approve()
        a.commit()
        """
    )
    assert out == ["ran"]


def test_duplicate_approval_in_requires_is_a_syntax_error():
    from sense_lang.errors import SenseSyntaxError

    with pytest.raises(SenseSyntaxError):
        run("irreversible action f() requires approval, approval:\n    print(1)\n")


def test_two_capabilities_in_requires_is_a_syntax_error():
    from sense_lang.errors import SenseSyntaxError

    with pytest.raises(SenseSyntaxError):
        run("irreversible action f() requires a.b, c.d:\n    print(1)\n")


def test_approve_free_function_form_redirects_regardless_of_argument():
    # Same reasoning as
    # test_action_policy.test_verify_free_function_form_redirects_regardless_of_argument:
    # approve(...) used to be a free function, now `.approve()` is a
    # method reachable only on an already-confirmed Action, so the old
    # name is a pure redirect regardless of what's passed.
    with pytest.raises(SenseRuntimeError, match="use action.approve"):
        run("x = 5\napprove(x)\n")


def test_approve_after_commit_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            irreversible action f() requires approval:
                print(1)
            a = f()
            a.verify()
            a.approve()
            a.commit()
            a.approve()
            """
        )


def test_ask_human_composes_with_approve_inside_agent():
    out = run(
        """
        irreversible action delete_account(user: String) requires approval:
            print("deleting " + user)

        a = delete_account("bob")
        a.verify()

        agent reviewer:
            answer = ask_human("approve?")
            if answer == "yes":
                a.approve()

        start(reviewer)
        resume(reviewer, "yes")
        a.commit()
        """
    )
    assert out == ["deleting bob"]


# -- audit log --------------------------------------------------------------------


def test_prepare_records_an_audit_entry():
    interp = run_interp('reversible action f():\n    print(1)\nrollback:\n    print(1)\na = f()\n')
    assert len(interp.audit_log) == 1
    entry = interp.audit_log[0]
    assert entry.action_name == "f"
    assert entry.kind == "reversible"
    assert entry.event == "prepared"


def test_full_lifecycle_records_three_entries():
    interp = run_interp(
        """
        irreversible action f():
            print(1)
        a = f()
        a.verify()
        a.commit()
        """
    )
    events = [e.event for e in interp.audit_log]
    assert events == ["prepared", "verified", "committed"]


def test_capability_denial_recorded_with_detail():
    interp = Interpreter(stdout=lambda _: None)
    try:
        interp.run_source(
            dedent(
                """
                irreversible action f() requires x.y:
                    print(1)
                policy:
                    deny x.y
                a = f()
                a.verify()
                """
            )
        )
    except SensePolicyError:
        pass
    assert interp.audit_log[-1].event == "denied"
    assert "x.y" in interp.audit_log[-1].detail


def test_approval_denial_recorded():
    interp = Interpreter(stdout=lambda _: None)
    try:
        interp.run_source(
            dedent(
                """
                irreversible action f() requires approval:
                    print(1)
                a = f()
                a.verify()
                a.commit()
                """
            )
        )
    except SenseApprovalError:
        pass
    assert interp.audit_log[-1].event == "denied"
    assert "approval" in interp.audit_log[-1].detail


def test_failed_commit_recorded():
    interp = Interpreter(stdout=lambda _: None)
    try:
        interp.run_source(
            dedent(
                """
                irreversible action f():
                    x = 1 / 0
                a = f()
                a.verify()
                a.commit()
                """
            )
        )
    except SenseRuntimeError:
        pass
    assert interp.audit_log[-1].event == "failed"
    assert "division by zero" in interp.audit_log[-1].detail


def test_audit_log_builtin_returns_readable_strings():
    out = run(
        """
        irreversible action f():
            print(1)
        a = f()
        a.verify()
        a.commit()
        for line in audit_log():
            print(line)
        """
    )
    # first line is f()'s own print(1); the rest are the three audit lines
    assert out[0] == "1"
    assert len(out) == 4
    assert "prepared" in out[1]
    assert "verified" in out[2]
    assert "committed" in out[3]


def test_audit_log_is_empty_with_no_actions():
    interp = run_interp('x = 5\nprint(x)\n')
    assert interp.audit_log == []


def test_audit_log_accumulates_across_multiple_actions():
    interp = run_interp(
        """
        irreversible action f():
            print(1)
        irreversible action g():
            print(2)
        a = f()
        b = g()
        a.verify()
        b.verify()
        """
    )
    names = [e.action_name for e in interp.audit_log]
    assert names == ["f", "g", "f", "g"]
