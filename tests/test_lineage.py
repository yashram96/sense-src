"""Execution lineage: each `AuditEntry` records which agent/tool/action
call boundaries it happened inside of, outermost-to-innermost, so the
Inspector's audit log can show a call tree instead of a flat list.
Built on `Environment.call_frame`/`get_lineage()` -- the same
scope-chain-walked-override mechanism `.agent`/`.model`/`.policy`
already use, and for the same reason (see environment.py's module
docstring): a shared mutable stack would corrupt itself the moment two
independent executions (a paused agent and its resumer, or two
`async agent`s) are both in flight.
"""

from __future__ import annotations

from textwrap import dedent

from sense_lang.interpreter import Interpreter


def run(source: str) -> Interpreter:
    interpreter = Interpreter(stdout=lambda _s: None)
    interpreter.run_source(dedent(source))
    return interpreter


def test_plain_top_level_call_has_no_lineage():
    interp = run(
        """
        tool search(query: String) returns String:
            return "results for " + query
        print(search("sense"))
        """
    )
    called = [e for e in interp.audit_log if e.kind == "tool" and e.event == "called"]
    assert len(called) == 1
    assert called[0].lineage == []
    assert " (via " not in called[0].describe()


def test_tool_called_from_inside_an_agent_body_shows_agent_in_lineage():
    interp = run(
        """
        tool search(query: String) returns String:
            return "results for " + query
        agent researcher:
            result = search("sense")
        start(researcher)
        """
    )
    called = [e for e in interp.audit_log if e.kind == "tool" and e.event == "called"]
    assert len(called) == 1
    assert called[0].lineage == ["agent:researcher"]
    assert "(via agent:researcher)" in called[0].describe()


def test_tool_called_from_inside_another_tools_body_shows_tool_in_lineage():
    interp = run(
        """
        tool inner(query: String) returns String:
            return "results for " + query
        tool outer(query: String) returns String:
            return inner(query)
        print(outer("sense"))
        """
    )
    called = [e for e in interp.audit_log if e.kind == "tool" and e.event == "called"]
    assert len(called) == 2
    outer_call = next(e for e in called if e.action_name == "outer")
    inner_call = next(e for e in called if e.action_name == "inner")
    assert outer_call.lineage == []
    assert inner_call.lineage == ["tool:outer"]


def test_tool_called_from_inside_an_action_body_shows_action_in_lineage():
    interp = run(
        """
        tool notify(who: String) returns String:
            return "notified " + who
        reversible action send_report(who: String) requires report.send:
            notify(who)
        rollback:
            pass_through = who
        policy:
            allow report.send
        a = send_report("alice")
        a.verify()
        a.commit()
        """
    )
    tool_called = [e for e in interp.audit_log if e.kind == "tool" and e.event == "called"]
    assert len(tool_called) == 1
    assert tool_called[0].lineage == ["reversible:send_report"]

    # The action's own audit events (prepared/verified/committed) happen
    # at the *caller's* scope, not inside its own body -- so they carry
    # no lineage of their own.
    prepared = next(e for e in interp.audit_log if e.event == "prepared")
    assert prepared.lineage == []


def test_describe_suffix_absent_when_lineage_empty_present_when_not():
    interp = run(
        """
        tool inner(query: String) returns String:
            return query
        tool outer(query: String) returns String:
            return inner(query)
        print(outer("sense"))
        """
    )
    called = [e for e in interp.audit_log if e.kind == "tool" and e.event == "called"]
    outer_call = next(e for e in called if e.action_name == "outer")
    inner_call = next(e for e in called if e.action_name == "inner")
    assert " (via " not in outer_call.describe()
    assert "(via tool:outer)" in inner_call.describe()
