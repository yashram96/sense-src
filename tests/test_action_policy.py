"""Phase 4 core slice: Action (prepare -> verify -> commit) and policy.

Central law under test: preparing an action must never cause its
external effect -- only .commit() may.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SensePolicyError, SenseRuntimeError, SenseTypeError
from sense_lang.interpreter import Interpreter


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


# -- prepare must not execute --------------------------------------------------


def test_calling_an_action_does_not_run_its_body():
    out = run(
        """
        irreversible action send(x: Int):
            print("side effect")
        a = send(1)
        print("no side effect yet")
        """
    )
    assert out == ["no side effect yet"]


def test_commit_runs_the_body_exactly_once():
    out = run(
        """
        irreversible action send(x: Int):
            print("sent " + str(x))
        a = send(5)
        a.verify()
        a.commit()
        """
    )
    assert out == ["sent 5"]


def test_action_returns_a_value_from_commit():
    out = run(
        """
        irreversible action compute(x: Int):
            return x * 2
        a = compute(21)
        a.verify()
        print(a.commit())
        """
    )
    assert out == ["42"]


# -- lifecycle / state machine -------------------------------------------------


def test_action_states_progress_prepared_verified_committed():
    out = run(
        """
        irreversible action noop():
            print("ran")
        a = noop()
        print(a.state)
        a.verify()
        print(a.state)
        a.commit()
        print(a.state)
        """
    )
    assert out == ["prepared", "verified", "ran", "committed"]


def test_commit_without_verify_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            irreversible action noop():
                print("ran")
            a = noop()
            a.commit()
            """
        )


def test_double_commit_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            irreversible action noop():
                print("ran")
            a = noop()
            a.verify()
            a.commit()
            a.commit()
            """
        )


def test_double_verify_raises():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            irreversible action noop():
                print("ran")
            a = noop()
            a.verify()
            a.verify()
            """
        )


def test_verify_free_function_form_redirects_regardless_of_argument():
    # verify(...) used to be a free function taking the action as an
    # argument -- expects an Action, got Int -- and this test proved it
    # rejected a bad argument type. Now `.verify()` is a method that can
    # only ever be reached on a value already known to be an Action (the
    # isinstance check lives one level up, at member-access dispatch), so
    # the old free-function name is a pure redirect: it fires regardless
    # of what's passed, since there's no argument to type-check any more.
    with pytest.raises(SenseRuntimeError, match="use action.verify"):
        run("verify(5)\n")


def test_type_of_action():
    out = run("irreversible action noop():\n    print(1)\na = noop()\nprint(type_of(a))\n")
    assert out == ["Action"]


def test_action_kind_readable():
    out = run(
        "irreversible action f():\n    print(1)\na = f()\nprint(a.kind)\n"
    )
    assert out == ["irreversible"]


# -- typed params -----------------------------------------------------------------


def test_action_param_type_mismatch_raises_at_call():
    with pytest.raises(SenseTypeError):
        run(
            """
            irreversible action f(x: Int):
                print(x)
            f("nope")
            """
        )


# -- policy: default allow, explicit allow/deny ----------------------------------


def test_action_with_no_requires_clause_needs_no_policy():
    out = run(
        """
        irreversible action f():
            print("ran")
        a = f()
        a.verify()
        a.commit()
        """
    )
    assert out == ["ran"]


def test_capability_defaults_to_allowed_when_unconfigured():
    out = run(
        """
        irreversible action f() requires some.capability:
            print("ran")
        a = f()
        a.verify()
        a.commit()
        """
    )
    assert out == ["ran"]


def test_policy_deny_blocks_verify():
    with pytest.raises(SensePolicyError):
        run(
            """
            irreversible action f() requires x.y:
                print("ran")
            policy:
                deny x.y
            a = f()
            a.verify()
            """
        )


def test_policy_allow_permits_commit():
    out = run(
        """
        irreversible action f() requires x.y:
            print("ran")
        policy:
            allow x.y
        a = f()
        a.verify()
        a.commit()
        """
    )
    assert out == ["ran"]


def test_policy_inline_single_rule_form():
    out = run(
        """
        irreversible action f() requires x.y:
            print("ran")
        policy: allow x.y
        a = f()
        a.verify()
        a.commit()
        """
    )
    assert out == ["ran"]


def test_policy_deny_after_verify_still_blocks_commit():
    # .commit() must independently re-check, not trust a cached .verify()
    # result.
    with pytest.raises(SensePolicyError):
        run(
            """
            irreversible action f() requires x.y:
                print("ran")
            policy:
                allow x.y
            a = f()
            a.verify()
            policy:
                deny x.y
            a.commit()
            """
        )


def test_session_scoped_policy_does_not_leak():
    out = run(
        """
        irreversible action f() requires x.y:
            print("ran")
        policy:
            deny x.y

        session s:
            policy:
                allow x.y
            a = f()
            a.verify()
            a.commit()
        """
    )
    assert out == ["ran"]


def test_session_scoped_policy_reverts_after_session_ends():
    with pytest.raises(SensePolicyError):
        run(
            """
            irreversible action f() requires x.y:
                print("ran")
            policy:
                deny x.y

            session s:
                policy:
                    allow x.y

            b = f()
            b.verify()
            """
        )


def test_agent_has_its_own_policy_scope():
    out = run(
        """
        irreversible action f() requires x.y:
            print("ran")
        policy:
            deny x.y

        agent a:
            policy:
                allow x.y
            act = f()
            act.verify()
            act.commit()

        start(a)
        """
    )
    assert out == ["ran"]


# -- rollback: / .rollback() (reversible-action compensation) --


def test_rollback_runs_the_rollback_body_with_the_same_bound_args():
    out = run(
        """
        reversible action update(user: String) requires profile.write:
            print("applying " + user)
        rollback:
            print("reverting " + user)

        policy:
            allow profile.write

        a = update("alice")
        a.verify()
        a.commit()
        a.rollback()
        """
    )
    assert out == ["applying alice", "reverting alice"]


def test_rollback_sees_local_variables_the_commit_body_computed():
    # Not just the action's own args -- rollback: is conceptually "undo
    # what commit just did," which routinely needs a value commit itself
    # computed (an inserted row's id, here just an incremented counter),
    # not only the original call's arguments.
    out = run(
        """
        reversible action register(name: String) requires users.write:
            new_id = 5 + 1
            print("registered " + name + " as #" + str(new_id))
        rollback:
            print("removing " + name + " (#" + str(new_id) + ")")

        policy:
            allow users.write

        a = register("ada")
        a.verify()
        a.commit()
        a.rollback()
        """
    )
    assert out == ["registered ada as #6", "removing ada (#6)"]


def test_rollback_returns_the_rollback_bodys_return_value():
    out = run(
        """
        reversible action update(user: String) requires profile.write:
            return "applied " + user
        rollback:
            return "reverted " + user

        policy:
            allow profile.write

        a = update("alice")
        a.verify()
        a.commit()
        print(a.rollback())
        """
    )
    assert out == ["reverted alice"]


def test_rollback_moves_state_to_rolled_back():
    out = run(
        """
        reversible action update(user: String) requires profile.write:
            pass_through = user
        rollback:
            pass_through = user

        policy:
            allow profile.write

        a = update("alice")
        a.verify()
        a.commit()
        a.rollback()
        print(a.state)
        """
    )
    assert out == ["rolled_back"]


def test_cannot_rollback_an_irreversible_action():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            irreversible action send(x: String):
                print(x)
            a = send("x")
            a.verify()
            a.commit()
            a.rollback()
            """
        )


def test_reversible_action_with_no_rollback_block_is_a_syntax_error():
    from sense_lang.errors import SenseSyntaxError

    with pytest.raises(SenseSyntaxError, match="must declare a 'rollback' block"):
        run(
            """
            reversible action update(x: String):
                print(x)
            """
        )


def test_cannot_rollback_before_commit():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            reversible action update(x: String):
                print(x)
            rollback:
                print(x)
            a = update("x")
            a.rollback()
            """
        )


def test_cannot_rollback_twice():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            reversible action update(x: String):
                print(x)
            rollback:
                print(x)
            a = update("x")
            a.verify()
            a.commit()
            a.rollback()
            a.rollback()
            """
        )


def test_rollback_denied_by_policy_leaves_action_committed():
    with pytest.raises(SensePolicyError):
        run(
            """
            reversible action update(x: String) requires profile.write:
                print(x)
            rollback:
                print(x)

            policy:
                allow profile.write

            a = update("x")
            a.verify()
            a.commit()
            policy:
                deny profile.write
            a.rollback()
            """
        )


def test_rollback_free_function_form_redirects_regardless_of_argument():
    # Same reasoning as test_verify_free_function_form_redirects_regardless_of_argument.
    with pytest.raises(SenseRuntimeError, match="use action.rollback"):
        run("rollback(1)\n")


def test_rollback_on_irreversible_action_is_a_syntax_error():
    from sense_lang.errors import SenseSyntaxError

    with pytest.raises(SenseSyntaxError):
        run(
            """
            irreversible action send(x: String):
                print(x)
            rollback:
                print(x)
            """
        )


def test_old_compensate_syntax_raises_a_helpful_error():
    from sense_lang.errors import SenseSyntaxError

    with pytest.raises(SenseSyntaxError, match="'rollback'"):
        run(
            """
            reversible action update(x: String):
                print(x)
            compensate:
                print(x)
            """
        )
