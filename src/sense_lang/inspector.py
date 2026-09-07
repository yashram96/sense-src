"""`sense inspect`: a live console over a running Sense program's declared
capability surface -- tools, actions, agents, memory, skills, and MCP
servers.

The design principle throughout: this file adds almost no new execution
logic. Every "try it out" action in the console is a thin wrapper around a
method `Interpreter` already has (`_call_tool`, `_prepare_action`,
`_verify_action`, `_commit_action`, `_rollback_action`, `_start_agent`,
`_resume_agent`, the `_pmem_*` family, `Environment.get_capability`) -- the
same enforcement path a hand-written Sense call goes through. This module's
own job is strictly discovery (walking a module's declared surface) and
transport (HTTP <-> those existing methods), not re-implementing anything
the interpreter already does.

One piece of state genuinely belongs here rather than on `Interpreter`:
each "Prepare" click on an action creates a new `SenseAction` instance not
tied to any name (the same action can be prepared many times), so
`InspectorState.live_actions` holds those, keyed by a generated id, scoped
to this console's own session -- lost on restart, which is an accepted v1
simplification for a local dev tool.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .environment import Environment
from .errors import SenseError, SenseRuntimeError
from .interpreter import Interpreter
from .values import (
    SenseAction,
    SenseActionDef,
    SenseAgent,
    SenseMcpServer,
    SensePersistentMemory,
    SenseSkill,
    SenseToolDef,
    stringify,
)

STATIC_DIR = Path(__file__).parent / "inspector_static"


def _to_jsonable(value: Any) -> Any:
    """Every Sense value already IS a JSON-safe Python value (int/float/
    str/bool/None/list of those) except the handful of wrapper types
    (Agent, Memory, a Python interop value, ...) -- those fall back to the
    same `stringify()` every `print()` already uses."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return stringify(value)


def _coerce_args(params: list, raw_args: list) -> list[Any]:
    """JSON has one number type; Sense (via Python) has two. A `Float`
    parameter sent as a whole number decodes from JSON as a Python `int`,
    which would fail `matches_type`'s Float check -- this is a JSON
    transport-boundary translation, not a language change, so it belongs
    here rather than in the interpreter."""
    coerced = []
    for param, arg in zip(params, raw_args):
        if param.type_ann is not None and param.type_ann.name == "Float" and isinstance(arg, int) and not isinstance(arg, bool):
            coerced.append(float(arg))
        else:
            coerced.append(arg)
    return coerced


def _param_spec(interp: Interpreter, param: Any) -> dict:
    return {
        "name": param.name,
        "type": str(param.type_ann) if param.type_ann is not None else None,
        "schema": interp._type_to_json_schema(param.type_ann),
    }


def _tool_spec(interp: Interpreter, name: str, tool: SenseToolDef) -> dict:
    return {
        "kind": "tool",
        "name": name,
        "async": tool.decl.is_async,
        "description": tool.decl.description,
        "doc": tool.doc,
        "requires": tool.decl.requires_capability,
        "returns": str(tool.decl.return_type) if tool.decl.return_type is not None else None,
        "params": [_param_spec(interp, p) for p in tool.decl.params],
    }


def _action_spec(interp: Interpreter, name: str, action_def: SenseActionDef) -> dict:
    return {
        "kind": action_def.kind,  # "reversible" | "irreversible"
        "name": name,
        "doc": action_def.doc,
        "requires": action_def.decl.requires_capability,
        "requires_approval": action_def.decl.requires_approval,
        "has_rollback": action_def.decl.rollback_body is not None,
        "params": [_param_spec(interp, p) for p in action_def.decl.params],
    }


def _agent_spec(name: str, agent: SenseAgent) -> dict:
    return {
        "kind": "agent",
        "name": name,
        "async": agent.decl.is_async,
        "status": agent.status,
        "pause_reason": agent.pause_reason,
        "doc": agent.doc,
    }


def _memory_spec(name: str, mem: Any) -> dict:
    if isinstance(mem, SensePersistentMemory):
        return {"kind": "persistent_memory", "name": name, "path": mem.path}
    return {"kind": "memory", "name": name, "hint": mem.kind, "keys": list(mem.store.keys())}


def _skill_spec(name: str, skill: SenseSkill) -> dict:
    return {"kind": "skill", "name": name, "description": skill.description, "tools": [t.name for t in skill.tools]}


def _mcp_spec(name: str, server: SenseMcpServer) -> dict:
    return {
        "kind": "mcp_server",
        "name": name,
        "capability": server.capability,
        "tools": [t.name for t in server.tools],
    }


def _key_declared(items: list, get_name) -> dict[str, Any]:
    """Groups a flat, declaration-ordered list (`Interpreter.declared_*`)
    by name. A name declared exactly once -- the overwhelming common case,
    true of every example that predates this -- keeps its plain name as
    the key, so nothing that already addresses a tool/action/agent/memory
    by its bare name (existing tests, docs, the console's own JS) needs to
    change. A name declared more than once -- a fresh `agent` per loop
    iteration, the exact case that used to be invisible entirely -- gets
    each instance its own `'<name>#<n>'` key instead, 1-indexed in
    declaration order, so every instance stays individually addressable
    rather than later ones silently shadowing the first.
    `build_surface` uses this to build each panel's dict; `InspectorState`
    calls it again with the identical `items`/`get_name` to resolve a key
    back to a value, so the two are always in sync."""
    by_name: dict[str, list] = {}
    for item in items:
        by_name.setdefault(get_name(item), []).append(item)
    keyed: dict[str, Any] = {}
    for name, instances in by_name.items():
        if len(instances) == 1:
            keyed[name] = instances[0]
        else:
            for i, instance in enumerate(instances, start=1):
                keyed[f"{name}#{i}"] = instance
    return keyed


def build_surface(module_env: Environment, interp: Interpreter) -> dict:
    """Classifies every tool/action/agent/memory/skill/MCP server ever
    declared -- from `Interpreter.declared_*` (appended at declaration
    time, regardless of which scope declared it), not by walking
    `module_env.vars`. Walking only the top-level scope used to miss
    anything declared inside a `session`/`for`/`agent` body (thrown away
    once that block finished) or inside an imported module (its own
    separate namespace) even though it genuinely ran -- this can't miss
    those, since it never depends on a name still being bound anywhere."""
    tools: dict[str, dict] = {}
    actions: dict[str, dict] = {}
    agents: dict[str, dict] = {}
    memories: dict[str, dict] = {}
    skills: dict[str, dict] = {}
    mcp_servers: dict[str, dict] = {}
    capabilities: set[str] = set()

    for key, tool in _key_declared(interp.declared_tools, lambda t: t.name).items():
        tools[key] = _tool_spec(interp, tool.name, tool)
        if tool.decl.requires_capability:
            capabilities.add(tool.decl.requires_capability)
    for key, action in _key_declared(interp.declared_actions, lambda a: a.name).items():
        actions[key] = _action_spec(interp, action.name, action)
        if action.decl.requires_capability:
            capabilities.add(action.decl.requires_capability)
    for key, agent in _key_declared(interp.declared_agents, lambda a: a.name).items():
        agents[key] = _agent_spec(agent.name, agent)
    for key, mem in _key_declared(interp.declared_memories, lambda m: m.name).items():
        memories[key] = _memory_spec(mem.name, mem)
    for key, skill in _key_declared(interp.declared_skills, lambda s: s.name).items():
        skills[key] = _skill_spec(skill.name, skill)
    for key, server in _key_declared(interp.declared_mcp_servers, lambda s: s.name).items():
        mcp_servers[key] = _mcp_spec(server.name, server)
        if server.capability:
            capabilities.add(server.capability)

    policy = {cap: module_env.get_capability(cap) for cap in sorted(capabilities)}

    return {
        "tools": tools,
        "actions": actions,
        "agents": agents,
        "memories": memories,
        "skills": skills,
        "mcp_servers": mcp_servers,
        "policy": policy,
        "dry_run": interp.dry_run,
    }


class InspectorState:
    """Holds the one `Interpreter`/`module_env` this console is inspecting,
    plus the console's own bookkeeping for prepared-but-not-yet-settled
    action instances. Every method here either reads existing interpreter
    state or calls straight into an existing `Interpreter` method -- see
    the module docstring."""

    def __init__(self, interpreter: Interpreter, module_env: Environment):
        self.interpreter = interpreter
        self.module_env = module_env
        self.live_actions: dict[str, SenseAction] = {}

    def surface(self) -> dict:
        return build_surface(self.module_env, self.interpreter)

    def audit_log(self) -> list[dict]:
        return [
            {
                "action_name": e.action_name,
                "kind": e.kind,
                "event": e.event,
                "detail": e.detail,
                "line": e.line,
                "timestamp": e.timestamp,
                "lineage": e.lineage,
            }
            for e in reversed(self.interpreter.audit_log)
        ]

    def _get_tool(self, key: str) -> SenseToolDef:
        tool = _key_declared(self.interpreter.declared_tools, lambda t: t.name).get(key)
        if tool is None:
            raise SenseRuntimeError(f"no tool named '{key}'")
        return tool

    def _get_action_def(self, key: str) -> SenseActionDef:
        action_def = _key_declared(self.interpreter.declared_actions, lambda a: a.name).get(key)
        if action_def is None:
            raise SenseRuntimeError(f"no action named '{key}'")
        return action_def

    def _get_agent(self, key: str) -> SenseAgent:
        agent = _key_declared(self.interpreter.declared_agents, lambda a: a.name).get(key)
        if agent is None:
            raise SenseRuntimeError(f"no agent named '{key}'")
        return agent

    def _get_memory(self, key: str) -> Any:
        mem = _key_declared(self.interpreter.declared_memories, lambda m: m.name).get(key)
        if mem is None:
            raise SenseRuntimeError(f"no memory named '{key}'")
        return mem

    def _get_live_action(self, action_id: str) -> SenseAction:
        action = self.live_actions.get(action_id)
        if action is None:
            raise SenseRuntimeError(
                f"no prepared action instance '{action_id}' (the console may have restarted)"
            )
        return action

    def call_tool(self, key: str, raw_args: list[Any]) -> dict:
        tool = self._get_tool(key)
        args = _coerce_args(tool.decl.params, raw_args)
        result = self.interpreter._call_tool(tool, args, self.module_env, None)
        return {"result": _to_jsonable(result)}

    def prepare_action(self, key: str, raw_args: list[Any]) -> dict:
        action_def = self._get_action_def(key)
        args = _coerce_args(action_def.decl.params, raw_args)
        action = self.interpreter._prepare_action(action_def, args, None, self.module_env)
        action_id = str(uuid.uuid4())
        self.live_actions[action_id] = action
        return {"id": action_id, "state": action.state}

    def _with_state_on_error(self, action: SenseAction, fn) -> Any:
        """A denial/failure raises before returning normally (e.g.
        `_verify_action` sets `action.state = "denied"` *then* raises) --
        without this, the console would have no way to learn the action
        moved to a dead-end state, and would keep showing stale buttons
        for a state that no longer exists. Attaching the state the
        interpreter already computed onto the exception lets `_dispatch`
        forward it to the browser alongside the error message."""
        try:
            return fn()
        except SenseError as exc:
            exc.action_state = action.state  # type: ignore[attr-defined]
            raise

    def verify_action(self, action_id: str) -> dict:
        action = self._get_live_action(action_id)
        self._with_state_on_error(action, lambda: self.interpreter._verify_action(self.module_env, None, action))
        return {"id": action_id, "state": action.state}

    def commit_action(self, action_id: str) -> dict:
        action = self._get_live_action(action_id)
        result = self._with_state_on_error(action, lambda: self.interpreter._commit_action(action, self.module_env))
        return {"id": action_id, "state": action.state, "result": _to_jsonable(result)}

    def rollback_action(self, action_id: str) -> dict:
        action = self._get_live_action(action_id)
        self._with_state_on_error(action, lambda: self.interpreter._rollback_action(self.module_env, None, action))
        return {"id": action_id, "state": action.state}

    def start_agent(self, key: str) -> dict:
        agent = self._get_agent(key)
        self.interpreter._start_agent(agent)
        return _agent_spec(agent.name, agent)

    def resume_agent(self, key: str, value: Any) -> dict:
        agent = self._get_agent(key)
        self.interpreter._resume_agent(agent, value)
        return _agent_spec(agent.name, agent)

    def memory_detail(self, key: str) -> dict:
        mem = self._get_memory(key)
        if isinstance(mem, SensePersistentMemory):
            keys = self.interpreter._pmem_keys(mem)
            entries = {k: _to_jsonable(self.interpreter._pmem_recall(mem, k, None)) for k in keys}
            return {"kind": "persistent_memory", "name": mem.name, "path": mem.path, "entries": entries}
        entries = {k: _to_jsonable(v) for k, v in mem.store.items()}
        return {"kind": "memory", "name": mem.name, "hint": mem.kind, "entries": entries}

    def memory_history(self, key: str, history_key: str) -> dict:
        mem = self._get_memory(key)
        if not isinstance(mem, SensePersistentMemory):
            raise SenseRuntimeError(f"'{mem.name}' is not a persistent memory")
        return {"name": mem.name, "key": history_key, "history": self.interpreter._pmem_history(mem, history_key, None)}


def _make_handler(state: InspectorState) -> type[BaseHTTPRequestHandler]:
    class InspectorHandler(BaseHTTPRequestHandler):
        server_version = "SenseInspector/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quieter, one-line request log
            print(f"  {self.command} {self.path}")

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length == 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _dispatch(self, handler) -> None:
            try:
                result = handler()
                self._send_json(200, {"ok": True, **(result if isinstance(result, dict) else {"result": result})})
            except SenseError as exc:
                payload = {"ok": False, "error": type(exc).__name__, "message": exc.message}
                state = getattr(exc, "action_state", None)
                if state is not None:
                    payload["state"] = state
                self._send_json(200, payload)
            except Exception as exc:  # noqa: BLE001 - surfaced to the console, not lost
                self._send_json(500, {"ok": False, "error": type(exc).__name__, "message": str(exc)})

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming convention
            # Unquoted: a disambiguated surface key like "worker#2" (see
            # inspector.py's _key_declared) contains "#", which is a URL
            # *fragment* delimiter in a browser's own URL parsing -- the
            # JS side percent-encodes it (encodeURIComponent) before
            # building the request, so this is where it comes back off.
            parts = [urllib.parse.unquote(p) for p in self.path.split("/")]
            if self.path == "/" or self.path == "/index.html":
                self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
                return
            if self.path == "/api/surface":
                self._dispatch(state.surface)
                return
            if self.path == "/api/audit-log":
                self._dispatch(lambda: {"entries": state.audit_log()})
                return
            if len(parts) == 4 and parts[1:3] == ["api", "memory"]:
                name = parts[3]
                self._dispatch(lambda: state.memory_detail(name))
                return
            if len(parts) == 6 and parts[1:3] == ["api", "memory"] and parts[4] == "history":
                name, key = parts[3], parts[5]
                self._dispatch(lambda: state.memory_history(name, key))
                return
            self._send_json(404, {"ok": False, "error": "NotFound", "message": f"no route for GET {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            parts = [urllib.parse.unquote(p) for p in self.path.split("/")]
            body = self._read_json_body()

            if len(parts) == 5 and parts[1:3] == ["api", "tool"] and parts[4] == "call":
                name = parts[3]
                self._dispatch(lambda: state.call_tool(name, body.get("args", [])))
                return
            if len(parts) == 5 and parts[1:3] == ["api", "action"] and parts[4] == "prepare":
                name = parts[3]
                self._dispatch(lambda: state.prepare_action(name, body.get("args", [])))
                return
            if len(parts) == 5 and parts[1:3] == ["api", "action"] and parts[4] in ("verify", "commit", "rollback"):
                action_id, op = parts[3], parts[4]
                fn = {"verify": state.verify_action, "commit": state.commit_action, "rollback": state.rollback_action}[op]
                self._dispatch(lambda: fn(action_id))
                return
            if len(parts) == 5 and parts[1:3] == ["api", "agent"] and parts[4] == "start":
                name = parts[3]
                self._dispatch(lambda: state.start_agent(name))
                return
            if len(parts) == 5 and parts[1:3] == ["api", "agent"] and parts[4] == "resume":
                name = parts[3]
                self._dispatch(lambda: state.resume_agent(name, body.get("value")))
                return
            self._send_json(404, {"ok": False, "error": "NotFound", "message": f"no route for POST {self.path}"})

    return InspectorHandler


def create_inspector_server(path: str, port: int = 4300, dry_run: bool = False) -> ThreadingHTTPServer:
    """Builds and binds the server without serving -- split out from
    `run_inspector` so tests can bind an ephemeral port (`port=0`), read
    back the real port via `server.server_address[1]`, and shut it down
    cleanly, instead of blocking forever on `serve_forever()`."""
    interpreter = Interpreter(dry_run=dry_run)
    module_env = interpreter.run_file(path)
    state = InspectorState(interpreter, module_env)
    handler = _make_handler(state)
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def run_inspector(path: str, port: int = 4300, dry_run: bool = False) -> int:
    server = create_inspector_server(path, port, dry_run)
    bound_port = server.server_address[1]
    print(f"Sense Inspector — {path}")
    print(f"  http://127.0.0.1:{bound_port}")
    if dry_run:
        print("  DRY RUN — commits/rollbacks clicked in the console are simulated, not executed")
    print("  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()
    return 0
