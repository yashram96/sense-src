"""`sense inspect` — a live console over a running program's declared
capability surface. Almost none of this file's logic is new: `build_surface`
is pure discovery over `Interpreter.declared_*` (flat, declaration-ordered
lists appended regardless of which scope or file declared the thing --
see `_key_declared`), and every "try it out" RPC wrapper calls straight
into an existing `Interpreter` method. These tests exercise that thin
layer directly first (fast, no HTTP), then prove the real transport works
end to end against a real background server on an ephemeral port.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from textwrap import dedent

import pytest

from sense_lang.errors import SenseApprovalError, SensePolicyError, SenseRuntimeError
from sense_lang.inspector import InspectorState, build_surface, create_inspector_server
from sense_lang.interpreter import Interpreter


def run(source: str):
    interpreter = Interpreter()
    module_env = interpreter.run_source(dedent(source))
    return interpreter, module_env


# -- build_surface --------------------------------------------------------


def test_surface_discovers_a_tool():
    interp, env = run(
        """
        tool search(query: String) returns String requires web.search "search the web":
            return "results for " + query
        policy:
            allow web.search
        """
    )
    surface = build_surface(env, interp)
    assert "search" in surface["tools"]
    spec = surface["tools"]["search"]
    assert spec["requires"] == "web.search"
    assert spec["description"] == "search the web"
    assert spec["params"] == [{"name": "query", "type": "String", "schema": {"type": "string"}}]
    assert spec["returns"] == "String"
    assert surface["policy"] == {"web.search": True}
    assert spec["doc"] is None  # no docstring here -- distinct from `description` above


def test_surface_exposes_a_tools_docstring_distinct_from_its_description():
    interp, env = run(
        '''
        tool search(query: String) returns String requires web.search "search the web":
            """
            Searches the web for a query and returns the results.
            """
            return "results for " + query
        policy:
            allow web.search
        '''
    )
    spec = build_surface(env, interp)["tools"]["search"]
    assert spec["description"] == "search the web"
    assert spec["doc"] == "Searches the web for a query and returns the results."


def test_surface_discovers_an_action_with_rollback():
    interp, env = run(
        """
        reversible action update_profile(user: String) requires profile.write:
            print("set")
        rollback:
            print("revert")
        policy:
            deny profile.write
        """
    )
    surface = build_surface(env, interp)
    spec = surface["actions"]["update_profile"]
    assert spec["kind"] == "reversible"
    assert spec["has_rollback"] is True
    assert surface["policy"] == {"profile.write": False}


def test_surface_discovers_an_agent_and_a_memory():
    interp, env = run(
        """
        agent approver:
            pause("waiting")
        memory notes: "working"
        persistent memory ledger: "test_inspector_ledger.sense_memory.db"
        """
    )
    surface = build_surface(env, interp)
    assert surface["agents"]["approver"]["status"] == "created"
    assert surface["memories"]["notes"]["kind"] == "memory"
    assert surface["memories"]["ledger"]["kind"] == "persistent_memory"


def test_surface_discovers_an_agent_declared_inside_a_session():
    """Previously invisible: build_surface only walked the top-level
    module_env, and a `session`'s own child scope is thrown away once it
    finishes."""
    interp, env = run(
        """
        session handle:
            agent worker:
                print("hi")
            start(worker)
        """
    )
    surface = build_surface(env, interp)
    assert "worker" in surface["agents"]
    assert surface["agents"]["worker"]["status"] == "completed"


def test_surface_discovers_a_tool_declared_inside_an_imported_module(tmp_path):
    """Previously invisible: an imported module's tools live in that
    module's own separate namespace, never the importer's module_env."""
    (tmp_path / "lib.sns").write_text(
        dedent(
            """
            tool helper(x: String) returns String:
                return x
            """
        ),
        encoding="utf-8",
    )
    main_path = tmp_path / "main.sns"
    main_path.write_text(
        dedent(
            """
            import "./lib.sns" as lib
            print(lib.helper("x"))
            """
        ),
        encoding="utf-8",
    )
    interp = Interpreter()
    env = interp.run_file(str(main_path))
    surface = build_surface(env, interp)
    assert "helper" in surface["tools"]


def test_surface_disambiguates_agents_declared_more_than_once_by_name():
    """A fresh `agent` per loop iteration used to either not appear at all
    (nested inside the loop's own scope) or, if it somehow did, one name
    would silently shadow another. Each instance now gets its own
    'name#n' key, all individually visible and startable."""
    interp, env = run(
        """
        for x in [1, 2, 3]:
            agent worker:
                print("hi")
            start(worker)
        """
    )
    surface = build_surface(env, interp)
    assert set(surface["agents"].keys()) == {"worker#1", "worker#2", "worker#3"}
    for key, spec in surface["agents"].items():
        assert spec["name"] == "worker"  # the disambiguation lives only in the key
        assert spec["status"] == "completed"


def test_surface_keeps_a_single_declaration_under_its_plain_name():
    """The common case (every existing example) is completely unaffected
    by the disambiguation logic: one declaration, one plain-name key."""
    interp, env = run(
        """
        agent solo:
            print("hi")
        """
    )
    surface = build_surface(env, interp)
    assert set(surface["agents"].keys()) == {"solo"}


# -- InspectorState RPCs ----------------------------------------------------


def test_call_tool_success():
    interp, env = run(
        """
        tool add(a: Int, b: Int) returns Int:
            return a + b
        """
    )
    state = InspectorState(interp, env)
    result = state.call_tool("add", [2, 3])
    assert result == {"result": 5}


def test_call_tool_denied_capability_raises():
    interp, env = run(
        """
        tool search(query: String) requires web.search:
            return query
        policy:
            deny web.search
        """
    )
    state = InspectorState(interp, env)
    with pytest.raises(SensePolicyError):
        state.call_tool("search", ["x"])


def test_call_tool_unknown_name_raises():
    interp, env = run("x = 1")
    state = InspectorState(interp, env)
    with pytest.raises(SenseRuntimeError):
        state.call_tool("does_not_exist", [])


def test_call_tool_coerces_int_to_float_param():
    interp, env = run(
        """
        tool half(x: Float) returns Float:
            return x / 2.0
        """
    )
    state = InspectorState(interp, env)
    result = state.call_tool("half", [10])  # JSON int, param is Float
    assert result == {"result": 5.0}


def test_action_full_lifecycle():
    interp, env = run(
        """
        reversible action update_profile(user: String) requires profile.write:
            print("set " + user)
        rollback:
            print("revert " + user)
        policy:
            allow profile.write
        """
    )
    state = InspectorState(interp, env)

    prepared = state.prepare_action("update_profile", ["alice"])
    assert prepared["state"] == "prepared"
    action_id = prepared["id"]

    verified = state.verify_action(action_id)
    assert verified["state"] == "verified"

    committed = state.commit_action(action_id)
    assert committed["state"] == "committed"

    rolled_back = state.rollback_action(action_id)
    assert rolled_back["state"] == "rolled_back"


def test_commit_before_verify_raises():
    interp, env = run(
        """
        irreversible action send_mail(to: String) requires email.send:
            print("sending")
        policy:
            allow email.send
        """
    )
    state = InspectorState(interp, env)
    prepared = state.prepare_action("send_mail", ["a@b.com"])
    with pytest.raises(SenseRuntimeError):
        state.commit_action(prepared["id"])


def test_commit_requiring_approval_without_approve_raises():
    interp, env = run(
        """
        irreversible action delete_account(user: String) requires account.delete, approval:
            print("deleting")
        policy:
            allow account.delete
        """
    )
    state = InspectorState(interp, env)
    prepared = state.prepare_action("delete_account", ["bob"])
    state.verify_action(prepared["id"])
    with pytest.raises(SenseApprovalError):
        state.commit_action(prepared["id"])


def test_unknown_action_instance_id_raises():
    interp, env = run("x = 1")
    state = InspectorState(interp, env)
    with pytest.raises(SenseRuntimeError):
        state.verify_action("not-a-real-id")


def test_agent_start_and_resume():
    interp, env = run(
        """
        agent approver:
            amount = 5000
            pause("needs approval for " + str(amount))
            print("approved")
        """
    )
    state = InspectorState(interp, env)

    started = state.start_agent("approver")
    assert started["status"] == "paused"
    assert "needs approval" in started["pause_reason"]

    resumed = state.resume_agent("approver", None)
    assert resumed["status"] == "completed"


def test_double_start_agent_raises():
    interp, env = run(
        """
        agent quick:
            print("done")
        """
    )
    state = InspectorState(interp, env)
    state.start_agent("quick")
    with pytest.raises(SenseRuntimeError):
        state.start_agent("quick")


def test_memory_detail_session_memory():
    interp, env = run(
        """
        memory notes: "working"
        notes.remember("topic", "sense")
        """
    )
    state = InspectorState(interp, env)
    detail = state.memory_detail("notes")
    assert detail["entries"] == {"topic": "sense"}


def test_memory_detail_persistent_memory_and_history(tmp_path):
    db_path = str(tmp_path / "inspector_test.sense_memory.db")
    interp, env = run(
        f"""
        persistent memory ledger: "{db_path.replace(chr(92), '/')}"
        ledger.remember("name", "ada")
        ledger.remember("name", "ada lovelace")
        """
    )
    state = InspectorState(interp, env)
    detail = state.memory_detail("ledger")
    assert detail["entries"] == {"name": "ada lovelace"}

    history = state.memory_history("ledger", "name")
    assert len(history["history"]) == 2


def test_audit_log_reflects_tool_calls_newest_first():
    interp, env = run(
        """
        tool a() returns Int:
            return 1
        tool b() returns Int:
            return 2
        """
    )
    state = InspectorState(interp, env)
    state.call_tool("a", [])
    state.call_tool("b", [])
    entries = state.audit_log()
    assert entries[0]["action_name"] == "b"
    assert entries[1]["action_name"] == "a"


# -- real end-to-end HTTP round trip ----------------------------------------


@pytest.fixture
def live_server(tmp_path):
    sns_file = tmp_path / "inspector_e2e.sns"
    sns_file.write_text(
        dedent(
            """
            tool add(a: Int, b: Int) returns Int:
                return a + b

            irreversible action update_profile(user: String) requires profile.write:
                print("set " + user)

            policy:
                allow profile.write

            agent approver:
                pause("waiting for approval")
            """
        ),
        encoding="utf-8",
    )
    server = create_inspector_server(str(sns_file), port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _get(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(base_url + path, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post(base_url: str, path: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_e2e_serves_static_ui(live_server):
    with urllib.request.urlopen(live_server + "/", timeout=5) as resp:
        body = resp.read().decode("utf-8")
    assert "Sense Inspector" in body


def test_e2e_surface_and_tool_call(live_server):
    surface = _get(live_server, "/api/surface")
    assert surface["ok"] is True
    assert "add" in surface["tools"]

    result = _post(live_server, "/api/tool/add/call", {"args": [2, 3]})
    assert result == {"ok": True, "result": 5}


def test_e2e_action_lifecycle(live_server):
    prepared = _post(live_server, "/api/action/update_profile/prepare", {"args": ["alice"]})
    assert prepared["ok"] is True
    action_id = prepared["id"]

    verified = _post(live_server, f"/api/action/{action_id}/verify", {})
    assert verified["state"] == "verified"

    committed = _post(live_server, f"/api/action/{action_id}/commit", {})
    assert committed["state"] == "committed"


def test_e2e_agent_start_and_audit_log(live_server):
    started = _post(live_server, "/api/agent/approver/start", {})
    assert started["status"] == "paused"

    # Agent start/pause isn't itself an audited event (only tool/action/model
    # calls are) -- exercise a tool call too, to prove the log endpoint
    # reflects real interpreter state, not just that it's reachable.
    _post(live_server, "/api/tool/add/call", {"args": [1, 1]})

    audit = _get(live_server, "/api/audit-log")
    assert audit["ok"] is True
    assert len(audit["entries"]) >= 1


def test_e2e_unknown_route_is_404(live_server):
    try:
        urllib.request.urlopen(live_server + "/api/nope", timeout=5)
        assert False, "expected HTTPError"
    except urllib.error.HTTPError as exc:
        assert exc.code == 404


@pytest.fixture
def duplicate_agent_server(tmp_path):
    """A disambiguated key like 'worker#2' contains '#' -- a URL *fragment*
    delimiter in a browser's own URL parsing, not just an ordinary path
    character. This fixture exists specifically to prove the real HTTP
    round trip (JS's encodeURIComponent -> the wire -> urllib.parse.unquote
    server-side -> _key_declared) survives that, not just the in-process
    InspectorState methods (which never touch a URL at all)."""
    sns_file = tmp_path / "dup_agents.sns"
    sns_file.write_text(
        dedent(
            """
            for x in [1, 2]:
                agent worker:
                    print("hi")
            """
        ),
        encoding="utf-8",
    )
    server = create_inspector_server(str(sns_file), port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_e2e_starting_one_of_several_same_named_agents_by_encoded_key(duplicate_agent_server):
    surface = _get(duplicate_agent_server, "/api/surface")
    assert set(surface["agents"].keys()) == {"worker#1", "worker#2"}
    assert all(spec["status"] == "created" for spec in surface["agents"].values())

    started = _post(duplicate_agent_server, "/api/agent/" + urllib.parse.quote("worker#2", safe="") + "/start", {})
    assert started["ok"] is True
    assert started["name"] == "worker"
    assert started["status"] == "completed"

    surface_after = _get(duplicate_agent_server, "/api/surface")
    assert surface_after["agents"]["worker#1"]["status"] == "created"  # untouched
    assert surface_after["agents"]["worker#2"]["status"] == "completed"
