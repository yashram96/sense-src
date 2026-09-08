"""Tree-walking interpreter for the Sense deterministic core."""

from __future__ import annotations

import importlib
import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import ast_nodes as n
from .environment import Environment
from .errors import (
    SenseApprovalError,
    SenseError,
    SenseImportError,
    SensePolicyError,
    SenseRuntimeError,
    SenseTypeError,
)
from .lexer import tokenize
from .mcp_client import McpBridge
from .parser import parse
from .providers import AnthropicProvider, MockModelProvider, OpenAIProvider, ToolSpec
from .values import (
    Answer,
    BuiltinFunction,
    McpTool,
    PythonValue,
    SenseAction,
    SenseActionDef,
    SenseAgent,
    SenseFunction,
    SenseFuture,
    SenseMcpServer,
    SenseMemory,
    SenseModel,
    SenseModule,
    SensePersistentMemory,
    SenseSkill,
    SenseToolDef,
    from_sense_value,
    is_truthy,
    matches_type,
    stringify,
    to_sense_value,
    type_name,
)


class _Return(Exception):
    def __init__(self, value: Any):
        self.value = value


class _Break(Exception):
    pass


class _Continue(Exception):
    pass


@dataclass
class AuditEntry:
    """One event in an Action's (or Tool's, or a real Model's) life.
    Host-side bookkeeping, same spirit as `TestResult` — `Interpreter.audit_log`
    accumulates these; the `audit_log()` builtin exposes a simplified,
    human-readable String view of the same data to Sense programs
    (there's no structured record/map value type yet to hand back the
    full entry)."""

    action_name: str
    kind: str  # "reversible" | "irreversible" | "tool" | "model"
    event: str  # "prepared" | "verified" | "committed" | "rolled_back" | "denied" | "failed" | "called"
    detail: str | None
    line: int | None
    timestamp: float
    lineage: list[str] = field(default_factory=list)  # e.g. ["agent:approver", "tool:search"] -- see Environment.get_lineage

    def describe(self) -> str:
        loc = f"[line {self.line}] " if self.line is not None else ""
        suffix = f" ({self.detail})" if self.detail else ""
        if self.kind in ("tool", "model"):
            subject = f"{self.kind} '{self.action_name}'"
        else:
            subject = f"{self.kind} action '{self.action_name}'"
        via = f" (via {' > '.join(self.lineage)})" if self.lineage else ""
        return f"{loc}{subject} -> {self.event}{suffix}{via}"


@dataclass
class TestResult:
    """One `test "description": ...` block's outcome. Not a Sense runtime
    value — host-side bookkeeping the CLI's `sense test` reads to print a
    summary. See `Interpreter._exec_TestStmt`."""

    description: str
    passed: bool
    error: str | None
    line: int


class Interpreter:
    def __init__(self, stdout=None, stdin=None, dry_run: bool = False):
        self.globals = Environment()
        self._stdout = stdout  # callable(str) -> None; defaults to print
        self._stdin = stdin  # callable(prompt: str) -> str; defaults to input()
        self.dry_run = dry_run  # see Interpreter._commit_action/_rollback_action
        self._install_builtins()
        self._module_cache: dict[str, SenseModule] = {}
        self._modules_in_progress: set[str] = set()
        self.test_results: list[TestResult] = []
        self.audit_log: list[AuditEntry] = []
        # Delegation and policy both live on Environment now, not here --
        # see environment.py's module docstring for why a single
        # Interpreter-global stack broke once `agent` needed pause/resume.

        # Every tool/action/agent/memory/skill/mcp server ever declared,
        # in declaration order, regardless of which scope declared it --
        # same "flat, Interpreter-level, appended at declaration time"
        # shape `self.audit_log` already used above, for the same reason:
        # `sense inspect`'s surface discovery (inspector.py's
        # build_surface) used to walk only the top-level module scope's
        # own vars, so anything declared inside a session/loop/agent body
        # (thrown away once that block finishes) or inside an imported
        # module (its own separate namespace) was invisible even though
        # it genuinely ran. These lists are what inspector.py now reads
        # instead -- discovery becomes "what was ever declared" rather
        # than "what's still bound at the top level right now."
        self.declared_tools: list[SenseToolDef] = []
        self.declared_actions: list[SenseActionDef] = []
        self.declared_agents: list[SenseAgent] = []
        self.declared_memories: list[Any] = []  # SenseMemory | SensePersistentMemory
        self.declared_skills: list[SenseSkill] = []
        self.declared_mcp_servers: list[SenseMcpServer] = []

    # -- public API ----------------------------------------------------------

    def run_source(self, source: str, file_path: str | None = None) -> Environment:
        tokens = tokenize(source)
        program = parse(tokens)
        module_env = self.globals.child()
        self._exec_block_statements(program.statements, module_env, file_path)
        return module_env

    def run_file(self, path: str) -> Environment:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        return self.run_source(source, file_path=os.path.abspath(path))

    # -- builtins -----------------------------------------------------------

    def _emit(self, text: str) -> None:
        if self._stdout is not None:
            self._stdout(text)
        else:
            print(text)

    def _read_line(self, prompt: str) -> str:
        if self._stdin is not None:
            return self._stdin(prompt)
        return input(prompt)

    def _install_builtins(self) -> None:
        def _print(*args: Any) -> None:
            self._emit(" ".join(stringify(a) for a in args))
            return None

        def _len(value: Any) -> int:
            if isinstance(value, (str, list)):
                return len(value)
            raise SenseRuntimeError(f"len() not supported for type {type_name(value)}")

        def _range(*args: int) -> list[int]:
            if len(args) == 1:
                return list(range(args[0]))
            if len(args) == 2:
                return list(range(args[0], args[1]))
            if len(args) == 3:
                return list(range(args[0], args[1], args[2]))
            raise SenseRuntimeError("range() expects 1 to 3 arguments")

        def _str(value: Any) -> str:
            return stringify(value)

        def _int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise SenseRuntimeError(f"cannot convert {value!r} to Int") from exc

        def _float(value: Any) -> float:
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise SenseRuntimeError(f"cannot convert {value!r} to Float") from exc

        def _type_of(value: Any) -> str:
            return type_name(value)

        def _push(arr: list, value: Any) -> list:
            if not isinstance(arr, list):
                raise SenseRuntimeError("push() expects an Array as its first argument")
            arr.append(value)
            return arr

        def _assert(*args: Any) -> None:
            if len(args) not in (1, 2):
                raise SenseRuntimeError("assert() expects 1 or 2 arguments")
            condition = args[0]
            message = args[1] if len(args) == 2 else "assertion failed"
            if not is_truthy(condition):
                raise SenseRuntimeError(message)
            return None

        def _mock_model_removed(*args: Any, **kwargs: Any) -> SenseModel:
            raise SenseRuntimeError(
                "mock_model(...) has been replaced -- use inference(\"mock\", ...) instead "
                "(e.g. inference(\"mock\", \"demo\", response_template: \"...\"))"
            )

        # `verify`/`approve`/`rollback` used to be free functions taking the
        # action as an argument -- inconsistent with `action.commit()`,
        # which was always a method. Now all four are `action.<name>()`.
        def _verify_removed(*args: Any, **kwargs: Any) -> SenseAction:
            raise SenseRuntimeError("verify(action) has been replaced -- use action.verify() instead")

        def _approve_removed(*args: Any, **kwargs: Any) -> SenseAction:
            raise SenseRuntimeError("approve(action) has been replaced -- use action.approve() instead")

        def _rollback_removed(*args: Any, **kwargs: Any) -> Any:
            raise SenseRuntimeError("rollback(action) has been replaced -- use action.rollback() instead")

        # provider name -> (ModelProvider subclass, default model_id, default api-key env var)
        _INFERENCE_PROVIDERS: dict[str, tuple[type, str, str]] = {
            "anthropic": (AnthropicProvider, "claude-sonnet-5", "ANTHROPIC_API_KEY"),
            "openai": (OpenAIProvider, "gpt-4o", "OPENAI_API_KEY"),
        }
        _INFERENCE_OPTIONS = {"temperature", "max_tokens", "api_key_env"}
        _MOCK_OPTIONS = {"response_template", "confidence"}

        def _inference(*args: Any, **kwargs: Any) -> SenseModel:
            """One function for every provider, real or offline -- the
            runtime, not a program, decides how reasoning is implemented,
            extended here to "and a program doesn't need a different
            function per vendor, or a separate function for an offline
            stand-in, either." Replaces the
            earlier `anthropic_model(...)`/`openai_model(...)` pair and
            the standalone `mock_model(...)`. No separate `name` argument
            for real providers -- `model NAME = inference(...)` is the
            intended way to name those, and the declaration already
            names it; called directly (`x = inference(...)`), the
            model_id itself is used as a reasonable default name.
            `inference("mock", ...)` is the one exception: there's no
            vendor model_id to default to, so its second positional
            argument is the name itself, and it needs no API key."""
            if not (1 <= len(args) <= 2):
                raise SenseRuntimeError(
                    "inference() expects (provider: String, model_id: String = default), "
                    "plus optional temperature:/max_tokens:/api_key_env: (real providers) "
                    "or response_template:/confidence: (\"mock\")"
                )
            provider_name = args[0]
            if not isinstance(provider_name, str):
                raise SenseRuntimeError("inference()'s provider argument must be a String")

            if provider_name == "mock":
                unknown = set(kwargs) - _MOCK_OPTIONS
                if unknown:
                    raise SenseRuntimeError(
                        f"inference(\"mock\"): unknown option(s) {', '.join(sorted(unknown))} -- "
                        f"supported: {', '.join(sorted(_MOCK_OPTIONS))}"
                    )
                name = args[1] if len(args) == 2 else "mock"
                if not isinstance(name, str):
                    raise SenseRuntimeError("inference(\"mock\")'s name argument must be a String")
                template = kwargs.get("response_template", "(mock reasoning about: {prompt})")
                if not isinstance(template, str):
                    raise SenseRuntimeError("inference(\"mock\")'s response_template option must be a String")
                confidence = kwargs.get("confidence", 0.5)
                if not isinstance(confidence, (int, float)):
                    raise SenseRuntimeError("inference(\"mock\")'s confidence option must be a number")
                return SenseModel(
                    name=name,
                    provider=MockModelProvider(response_template=template, confidence=float(confidence)),
                )

            spec = _INFERENCE_PROVIDERS.get(provider_name)
            if spec is None:
                raise SenseRuntimeError(
                    f"inference(): unknown provider '{provider_name}' -- supported: "
                    f"{', '.join(sorted([*_INFERENCE_PROVIDERS, 'mock']))}"
                )
            provider_cls, default_model_id, default_api_key_env = spec
            model_id = args[1] if len(args) == 2 else default_model_id
            if not isinstance(model_id, str):
                raise SenseRuntimeError("inference()'s model_id argument must be a String")

            unknown = set(kwargs) - _INFERENCE_OPTIONS
            if unknown:
                raise SenseRuntimeError(
                    f"inference(): unknown option(s) {', '.join(sorted(unknown))} -- "
                    f"supported: {', '.join(sorted(_INFERENCE_OPTIONS))}"
                )
            api_key_env = kwargs.get("api_key_env", default_api_key_env)
            if not isinstance(api_key_env, str):
                raise SenseRuntimeError("inference()'s api_key_env option must be a String")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise SenseRuntimeError(
                    f"inference() needs the '{api_key_env}' environment variable set to a real API key"
                )
            temperature = kwargs.get("temperature")
            if temperature is not None and not isinstance(temperature, (int, float)):
                raise SenseRuntimeError("inference()'s temperature option must be a number")
            max_tokens = kwargs.get("max_tokens")
            if max_tokens is not None and not isinstance(max_tokens, int):
                raise SenseRuntimeError("inference()'s max_tokens option must be an Int")

            provider = provider_cls(model_id=model_id, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
            return SenseModel(name=model_id, provider=provider)

        def _skill(*args: Any) -> SenseSkill:
            if len(args) != 3:
                raise SenseRuntimeError("skill() expects (name: String, description: String, tools: Array)")
            name, description, tools = args
            if not isinstance(name, str):
                raise SenseRuntimeError("skill()'s name argument must be a String")
            if not isinstance(description, str):
                raise SenseRuntimeError("skill()'s description argument must be a String")
            if not isinstance(tools, list):
                raise SenseRuntimeError(f"skill()'s tools argument must be an Array, got {type_name(tools)}")
            if not tools:
                raise SenseRuntimeError(f"skill '{name}' needs at least one tool")
            for t in tools:
                # Reused purely for its validation (is a tool, not async,
                # has a description) -- same checks ask_with_tools()
                # applies, just caught here, at skill-creation time, for
                # the earliest and clearest possible error.
                self._build_tool_spec(t, None)
            skill = SenseSkill(name=name, description=description, tools=list(tools))
            self.declared_skills.append(skill)
            return skill

        def _connect_mcp(*args: Any) -> SenseMcpServer:
            if not (3 <= len(args) <= 4):
                raise SenseRuntimeError(
                    "connect_mcp() expects (name: String, command: String, args: Array<String>, "
                    "capability: String = nil)"
                )
            name, command, cmd_args = args[0], args[1], args[2]
            capability = args[3] if len(args) == 4 else None
            if not isinstance(name, str):
                raise SenseRuntimeError("connect_mcp()'s name argument must be a String")
            if not isinstance(command, str):
                raise SenseRuntimeError("connect_mcp()'s command argument must be a String")
            if not isinstance(cmd_args, list) or not all(isinstance(a, str) for a in cmd_args):
                raise SenseRuntimeError("connect_mcp()'s args argument must be an Array of Strings")
            if capability is not None and not isinstance(capability, str):
                raise SenseRuntimeError("connect_mcp()'s capability argument must be a String")
            bridge = McpBridge(command, cmd_args)
            try:
                tool_infos = bridge.start()
            except Exception as exc:
                raise SenseRuntimeError(
                    f"connect_mcp() failed to connect to '{name}': {type(exc).__name__}: {exc}"
                ) from exc
            server = SenseMcpServer(name=name, capability=capability, tools=[], _bridge=bridge)
            server.tools = [
                McpTool(server=server, name=info.name, description=info.description, input_schema=info.input_schema)
                for info in tool_infos
            ]
            self.declared_mcp_servers.append(server)
            return server

        def _ask(env: Environment, line: int, prompt: Any) -> Answer:
            model = env.get_current_model()
            if model is None:
                raise SenseRuntimeError(
                    "ask() has no model configured — declare one (e.g. inference(\"mock\")) and "
                    "activate it with 'set delegation = <model>' first",
                    line,
                )
            return self._ask_model(model, prompt, env, line)

        def _ask_with_memory_builtin(env: Environment, line: int, prompt: Any, mem: Any) -> Answer:
            model = env.get_current_model()
            if model is None:
                raise SenseRuntimeError(
                    "ask_with_memory() has no model configured — declare one (e.g. inference(\"mock\")) "
                    "and activate it with 'set delegation = <model>' first",
                    line,
                )
            return self._ask_with_memory(model, prompt, mem, env, line)

        def _ask_with_tools_builtin(env: Environment, line: int, prompt: Any, tools: Any) -> Answer:
            model = env.get_current_model()
            if model is None:
                raise SenseRuntimeError(
                    "ask_with_tools() has no model configured — declare one "
                    "(e.g. model claude = inference(\"anthropic\")) and activate it with "
                    "'set delegation = <model>' first",
                    line,
                )
            if not isinstance(tools, list):
                raise SenseRuntimeError(
                    f"ask_with_tools() expects an Array of tools, got {type_name(tools)}", line
                )
            return self._ask_with_tools(model, prompt, tools, env, line)

        def _audit_log_view() -> list[str]:
            return [entry.describe() for entry in self.audit_log]

        # (arity, fn) for an ordinary builtin; (arity, fn, True) if it needs
        # (env, line) threaded in before its own args — see BuiltinFunction.
        builtins: dict[str, tuple] = {
            "print": (-1, _print),
            "len": (1, _len),
            "range": (-1, _range),
            "str": (1, _str),
            "int": (1, _int),
            "float": (1, _float),
            "type_of": (1, _type_of),
            "push": (2, _push),
            "assert": (-1, _assert),
            "mock_model": (-1, _mock_model_removed),
            "inference": (-1, _inference),
            "skill": (3, _skill),
            "connect_mcp": (-1, _connect_mcp),
            "ask": (1, _ask, True),
            "ask_with_memory": (2, _ask_with_memory_builtin, True),
            "ask_with_tools": (2, _ask_with_tools_builtin, True),
            "start": (1, self._start_agent),
            "verify": (-1, _verify_removed),
            "approve": (-1, _approve_removed),
            "pause": (-1, self._pause_agent, True),
            "resume": (-1, self._resume_agent),
            "send": (2, self._send_message),
            "receive": (-1, self._receive_message, True),
            "ask_human": (1, self._ask_human, True),
            "audit_log": (0, _audit_log_view),
            "rollback": (-1, _rollback_removed),
            "await": (1, self._await, True),
        }
        # The only builtin that accepts labeled call arguments today (e.g.
        # inference(..., temperature: 0.2)) -- see Call.kwargs/BuiltinFunction
        # .accepts_kwargs for why this stays a narrow, explicit opt-in
        # rather than a general keyword-argument mechanism.
        _kwargs_aware = {"inference"}
        for name, spec in builtins.items():
            arity, fn = spec[0], spec[1]
            needs_context = spec[2] if len(spec) > 2 else False
            self.globals.define(
                name,
                BuiltinFunction(
                    name=name, arity=arity, fn=fn, needs_context=needs_context,
                    accepts_kwargs=name in _kwargs_aware,
                ),
            )

    def _ask_model(
        self, model: SenseModel, prompt: Any, env: Environment, line: int | None = None
    ) -> Answer:
        if not isinstance(prompt, str):
            raise SenseRuntimeError(f"ask() expects a String prompt, got {type_name(prompt)}", line)
        # Only a real provider (e.g. AnthropicProvider) declares this --
        # MockModelProvider doesn't, so mock calls are never gated or
        # audited. A real call costs money and sends prompt data to a
        # third party, the same shape of thing tool/action's `requires`
        # already exists for.
        cap = getattr(model.provider, "capability", None)
        if cap is not None and not env.get_capability(cap):
            raise SensePolicyError(
                f"model '{model.name}' requires capability '{cap}', which is denied by policy", line
            )
        try:
            response = model.provider.complete(prompt)
        except Exception as exc:
            raise SenseRuntimeError(f"model '{model.name}' failed: {type(exc).__name__}: {exc}", line) from exc
        if cap is not None:
            self.audit_log.append(
                AuditEntry(
                    action_name=model.name, kind="model", event="called", detail=None, line=line,
                    timestamp=time.time(), lineage=env.get_lineage(),
                )
            )
        return Answer(value=response.text, confidence=response.confidence, source=model.name)

    def _memory_snapshot(self, mem: Any, line: int | None) -> dict[str, Any]:
        """Every current key/value pair in a `memory` or `persistent
        memory`, uniformly -- used to build `ask_with_memory`'s
        context. Both memory kinds already expose the same
        remember/recall/keys shape to Sense programs; this just reads
        that shape from the host side instead of through member access."""
        if isinstance(mem, SenseMemory):
            return dict(mem.store)
        if isinstance(mem, SensePersistentMemory):
            return {key: self._pmem_recall(mem, key, line) for key in self._pmem_keys(mem)}
        raise SenseRuntimeError(f"ask_with_memory() expects a Memory, got {type_name(mem)}", line)

    def _ask_with_memory(
        self, model: SenseModel, prompt: Any, mem: Any, env: Environment, line: int | None = None
    ) -> Answer:
        """Retrieval-into-prompt, the lightweight half of "memory wired
        into reasoning": the memory's entire current contents are
        injected as context ahead of the prompt, then handled by
        `_ask_model` exactly as `ask()` already is -- same
        capability gate (if the delegated provider is real), same error
        wrapping, works with `inference("mock", ...)` too, since it's
        just string composition ahead of an ordinary ask() call.

        Write-back (the model *storing* something back to memory) is
        deliberately not built here: it composes for free out of what
        already exists -- write a small `tool` wrapping
        `<memory>.remember(...)` and hand it to `ask_with_tools`,
        letting the model decide what's worth remembering, capability-
        gated and audited like any other tool call. An automatic
        write-back (guessing what to store from free-text output) isn't
        attempted here -- it would mean unreliably inventing facts to persist.
        """
        if not isinstance(prompt, str):
            raise SenseRuntimeError(f"ask_with_memory() expects a String prompt, got {type_name(prompt)}", line)
        snapshot = self._memory_snapshot(mem, line)
        if snapshot:
            known = "\n".join(f"- {key}: {stringify(value)}" for key, value in snapshot.items())
            combined_prompt = f"Known information from memory:\n{known}\n\n{prompt}"
        else:
            combined_prompt = prompt
        return self._ask_model(model, combined_prompt, env, line)

    _MAX_TOOL_CALL_ITERATIONS = 10

    def _type_to_json_schema(self, type_ann: "n.TypeAnnotation | None") -> dict:
        if type_ann is None:
            return {}
        primitive = {"Int": "integer", "Float": "number", "String": "string", "Bool": "boolean"}
        if type_ann.name in primitive:
            return {"type": primitive[type_ann.name]}
        if type_ann.name == "Array":
            schema: dict[str, Any] = {"type": "array"}
            if type_ann.params:
                schema["items"] = self._type_to_json_schema(type_ann.params[0])
            return schema
        return {}  # Any, or an unknown/custom name -- unconstrained, same as elsewhere in the type checker

    def _build_tool_spec(self, tool: Any, line: int | None) -> ToolSpec:
        if isinstance(tool, McpTool):
            # The server's own schema is used verbatim -- no Sense-side
            # type annotation to derive one from, unlike a hand-written
            # `tool`. Namespaced by server name so two servers (or a
            # server and a bare Sense tool) can't collide on a generic
            # tool name like "search" -- tools_by_name then resolves back
            # to this exact McpTool via that same qualified name.
            return ToolSpec(
                name=f"{tool.server.name}.{tool.name}",
                description=tool.description or "(no description provided)",
                parameters=tool.input_schema or {"type": "object", "properties": {}},
            )
        # Shared validation for ask_with_tools() and skill() -- the
        # wording deliberately doesn't name either caller, since either
        # can trigger it.
        if not isinstance(tool, SenseToolDef):
            raise SenseRuntimeError(
                f"only tools declared with 'tool' (or 'async tool') are accepted here, got {type_name(tool)}",
                line,
            )
        if tool.decl.is_async:
            raise SenseRuntimeError(
                f"tool '{tool.name}' is async -- async tools aren't supported here yet "
                "(its result is a pending Future, not a value)",
                line,
            )
        if not tool.description:
            raise SenseRuntimeError(
                f"tool '{tool.name}' has no description -- add one to expose it to a model, "
                f'e.g. \'tool {tool.name}(...) "what it does":\'',
                line,
            )
        properties = {p.name: self._type_to_json_schema(p.type_ann) for p in tool.decl.params}
        required = [p.name for p in tool.decl.params]
        return ToolSpec(
            name=tool.name,
            description=tool.description,
            parameters={"type": "object", "properties": properties, "required": required},
        )

    def _ask_with_tools(
        self, model: SenseModel, prompt: Any, tools: list[Any], env: Environment, line: int | None = None
    ) -> Answer:
        """A real model choosing which `tool` to invoke -- the
        Action/Tool capability model finally exercised against an
        autonomous decision, not just Sense code deciding by hand. The
        model proposes; `self._call_tool` still disposes -- every
        requested call goes through the exact same
        arity/capability/type-check/audit path a hand-written call
        reaches, so a policy-denied tool is never executed, model request
        or not."""
        if not isinstance(prompt, str):
            raise SenseRuntimeError(f"ask_with_tools() expects a String prompt, got {type_name(prompt)}", line)
        cap = getattr(model.provider, "capability", None)
        if cap is not None and not env.get_capability(cap):
            raise SensePolicyError(
                f"model '{model.name}' requires capability '{cap}', which is denied by policy", line
            )

        # `tools` may mix bare `tool` values with `skill(...)` bundles --
        # a skill flattens into its component tools, and its description
        # (not just documentation -- see values.py's SenseSkill) becomes
        # part of the prompt, same "not cosmetic" pattern as
        # ask_with_memory's context injection.
        tool_specs: list[ToolSpec] = []
        tools_by_name: dict[str, SenseToolDef] = {}
        skill_descriptions: list[str] = []
        for item in tools:
            item_tools = item.tools if isinstance(item, SenseSkill) else [item]
            if isinstance(item, SenseSkill):
                skill_descriptions.append(f"{item.name}: {item.description}")
            for t in item_tools:
                spec = self._build_tool_spec(t, line)
                if spec.name in tools_by_name:
                    raise SenseRuntimeError(
                        f"duplicate tool name '{spec.name}' across the given tools/skills", line
                    )
                tool_specs.append(spec)
                tools_by_name[spec.name] = t
        if skill_descriptions:
            preamble = "Available skills:\n" + "\n".join(f"- {d}" for d in skill_descriptions) + "\n\n"
            prompt = preamble + prompt

        messages: list[Any] = [{"role": "user", "content": prompt}]
        for _ in range(self._MAX_TOOL_CALL_ITERATIONS):
            try:
                turn = model.provider.complete_with_tools(messages, tool_specs)
            except NotImplementedError as exc:
                raise SenseRuntimeError(
                    f"model '{model.name}' does not support ask_with_tools(): {exc}", line
                ) from exc
            except Exception as exc:
                raise SenseRuntimeError(f"model '{model.name}' failed: {type(exc).__name__}: {exc}", line) from exc

            if turn.text is not None:
                if cap is not None:
                    self.audit_log.append(
                        AuditEntry(
                            action_name=model.name, kind="model", event="called", detail=None, line=line,
                            timestamp=time.time(), lineage=env.get_lineage(),
                        )
                    )
                return Answer(value=turn.text, confidence=None, source=model.name)

            messages.append(turn.raw_assistant_message)
            results: list[tuple[str, str]] = []
            for call in turn.tool_calls:
                tool = tools_by_name.get(call.name)
                if tool is None:
                    result_text = f"error: no such tool '{call.name}'"
                elif isinstance(tool, McpTool):
                    try:
                        # A parallel path to _call_tool, not through it --
                        # an MCP tool has no Sense body to execute -- but
                        # the same shape: capability check, audit, denial
                        # fed back rather than raised.
                        result_text = self._call_mcp_tool(tool, call.arguments, env, line)
                    except SenseError as exc:
                        result_text = f"error: {exc}"
                else:
                    args = [call.arguments.get(p.name) for p in tool.decl.params]
                    try:
                        # The same method a hand-written `tool_name(...)`
                        # call reaches -- a model's request gets no more
                        # trust than Sense code would. A denial here is
                        # still fully audited by _call_tool itself; it's
                        # only *re-raising* that's suppressed, so the
                        # model can respond instead of the whole call
                        # crashing.
                        result = self._call_tool(tool, args, env, line)
                        result_text = stringify(result)
                    except SenseError as exc:
                        result_text = f"error: {exc}"
                results.append((call.id, result_text))
            # Each vendor's own wire shape for "here's what those tools
            # returned" -- Anthropic bundles every result into one user
            # message, OpenAI wants one separate tool-role message per
            # call. See providers.py's ModelProvider.format_tool_results.
            messages.extend(model.provider.format_tool_results(results))

        raise SenseRuntimeError(
            f"model '{model.name}' did not produce a final answer within "
            f"{self._MAX_TOOL_CALL_ITERATIONS} tool-call iterations",
            line,
        )

    # -- statement execution ---------------------------------------------------

    def _exec_block_statements(self, statements: list[n.Node], env: Environment, file_path: str | None) -> None:
        for stmt in statements:
            self._execute(stmt, env, file_path)

    def _execute(self, stmt: n.Node, env: Environment, file_path: str | None) -> None:
        method = getattr(self, f"_exec_{type(stmt).__name__}", None)
        if method is None:
            raise SenseRuntimeError(f"no executor for statement {type(stmt).__name__}", stmt.line)
        method(stmt, env, file_path)

    def _exec_TypedDecl(self, stmt: n.TypedDecl, env: Environment, file_path: str | None) -> None:
        value = self._evaluate(stmt.value, env, file_path)
        if not matches_type(value, stmt.type_ann):
            raise SenseTypeError(
                f"cannot assign value of type {type_name(value)} to '{stmt.name}: {stmt.type_ann}'",
                stmt.line,
            )
        env.define_or_assign(stmt.name, value)

    def _exec_LocalDecl(self, stmt: n.LocalDecl, env: Environment, file_path: str | None) -> None:
        value = self._evaluate(stmt.value, env, file_path)
        if stmt.type_ann is not None and not matches_type(value, stmt.type_ann):
            raise SenseTypeError(
                f"cannot assign value of type {type_name(value)} to '{stmt.name}: {stmt.type_ann}'",
                stmt.line,
            )
        env.define(stmt.name, value)

    def _exec_SetStmt(self, stmt: n.SetStmt, env: Environment, file_path: str | None) -> None:
        value = self._evaluate(stmt.value, env, file_path)
        if stmt.name == "delegation":
            if not isinstance(value, SenseModel):
                raise SenseTypeError(
                    f"'set delegation' requires a Model value, got {type_name(value)}", stmt.line
                )
            env.set_current_model(value)
            return
        raise SenseRuntimeError(
            f"unknown 'set' target '{stmt.name}' (only 'delegation' is supported so far)", stmt.line
        )

    def _exec_ModelDecl(self, stmt: n.ModelDecl, env: Environment, file_path: str | None) -> None:
        value = self._evaluate(stmt.value, env, file_path)
        if not isinstance(value, SenseModel):
            raise SenseTypeError(f"'model {stmt.name}' expects a Model, got {type_name(value)}", stmt.line)
        # The declaration names it -- inference(...) doesn't need its own
        # separate name argument to be meaningful; whatever name the
        # expression's own value carried is overridden here.
        value.name = stmt.name
        env.define(stmt.name, value)

    def _exec_FnDecl(self, stmt: n.FnDecl, env: Environment, file_path: str | None) -> None:
        env.define(stmt.name, SenseFunction(decl=stmt, closure=env))

    def _exec_TestStmt(self, stmt: n.TestStmt, env: Environment, file_path: str | None) -> None:
        try:
            self._exec_block_statements(stmt.body.statements, env.child(), file_path)
            self.test_results.append(TestResult(stmt.description, True, None, stmt.line))
        except SenseError as exc:
            self.test_results.append(TestResult(stmt.description, False, str(exc), stmt.line))
        except (_Return, _Break, _Continue):
            raise  # not test failures -- genuine misuse (return/break/continue with no enclosing function/loop)
        except Exception as exc:  # noqa: BLE001 - an unexpected host-level error still counts as a test failure
            self.test_results.append(TestResult(stmt.description, False, f"{type(exc).__name__}: {exc}", stmt.line))

    def _exec_SessionStmt(self, stmt: n.SessionStmt, env: Environment, file_path: str | None) -> None:
        # A fresh child env is enough: `set delegation`/`policy` inside will
        # only ever mutate *this* env's own model/policy slot, and lookups
        # from inside walk up to the surrounding scope automatically — see
        # environment.py. Nothing to push/pop.
        self._exec_block_statements(stmt.body.statements, env.child(), file_path)

    def _exec_AgentDecl(self, stmt: n.AgentDecl, env: Environment, file_path: str | None) -> None:
        agent_env = env.child()
        agent = SenseAgent(name=stmt.name, decl=stmt, env=agent_env)
        agent_env.agent = agent  # so pause() inside the body can find its own agent
        env.define_or_assign(stmt.name, agent)
        self.declared_agents.append(agent)

    def _run_agent_body(self, agent: SenseAgent) -> None:
        """Runs on the agent's own thread — see SenseAgent's docstring for
        the cooperative hand-off design (exactly one of {caller, agent} is
        ever actually executing at a time)."""
        try:
            try:
                self._exec_block_statements(agent.decl.body.statements, agent.env, None)
            except _Return:
                pass  # a bare `return` just ends the body early, same as falling off the end
            if agent.status == "running":  # not paused/failed mid-way
                agent.status = "completed"
        except BaseException as exc:  # noqa: BLE001 - must capture to hand back to the caller's thread
            agent.status = "failed"
            agent._pending_error = exc
        finally:
            agent._settled_event.set()

    def _start_agent(self, agent: SenseAgent) -> SenseAgent:
        if not isinstance(agent, SenseAgent):
            raise SenseRuntimeError(f"start() expects an Agent, got {type_name(agent)}")
        if agent.status != "created":
            raise SenseRuntimeError(
                f"agent '{agent.name}' has already been started (status: {agent.status}) — "
                "an agent can only be started once in this version"
            )
        agent.status = "running"
        agent._resume_event = threading.Event()
        agent._settled_event = threading.Event()
        agent._thread = threading.Thread(target=self._run_agent_body, args=(agent,), daemon=True)
        agent._thread.start()
        if agent.decl.is_async:
            # Genuinely concurrent: the caller and this agent's body can now
            # both be executing at the same time. await(agent) blocks for
            # the result later.
            return agent
        agent._settled_event.wait()  # blocks until the agent pauses, completes, or fails
        return self._collect_agent_result(agent)

    def _pause_agent(self, env: Environment, line: int, *args: Any) -> Any:
        if len(args) > 1:
            raise SenseRuntimeError("pause() expects 0 or 1 arguments (an optional reason)", line)
        agent = env.get_current_agent()
        if agent is None:
            raise SenseRuntimeError("pause() can only be called inside a running agent's body", line)
        agent.pause_reason = args[0] if args else None
        agent.status = "paused"
        agent._settled_event.set()  # let start()/resume() (blocked, waiting) proceed
        agent._resume_event.wait()  # block THIS thread until resume() signals it
        agent._resume_event.clear()
        agent.status = "running"
        # Whatever resume(agent, value) passed -- nil if resume() was called
        # with no second argument. This is what makes ask_human() possible:
        # it's built on pause() carrying a value back in, not just a signal.
        value, agent._resume_value = agent._resume_value, None
        return value

    def _resume_agent(self, *args: Any) -> SenseAgent:
        if len(args) not in (1, 2):
            raise SenseRuntimeError("resume() expects 1 or 2 arguments (agent, value?)")
        agent, value = args[0], (args[1] if len(args) == 2 else None)
        if not isinstance(agent, SenseAgent):
            raise SenseRuntimeError(f"resume() expects an Agent, got {type_name(agent)}")
        if agent.status != "paused":
            raise SenseRuntimeError(
                f"cannot resume agent '{agent.name}' — status is '{agent.status}', not 'paused'"
            )
        agent._resume_value = value
        agent._settled_event.clear()
        agent._resume_event.set()
        if agent.decl.is_async:
            return agent
        agent._settled_event.wait()  # blocks until it pauses again, completes, or fails
        return self._collect_agent_result(agent)

    def _send_message(self, agent: Any, message: Any) -> None:
        """`send(agent, message)`: inter-agent communication, built on the
        same "reuse a stdlib primitive" call `SensePersistentMemory` made for its
        own thread-safety -- `SenseAgent._mailbox` is a `queue.Queue`,
        already safe for many concurrent senders (several `async agent`s
        all messaging one coordinator) without a new lock of Sense's own.
        Callable from anywhere -- top-level script code, another agent's
        body, even before the target has been `start()`-ed -- since the
        mailbox exists from the moment `agent name: ...` is declared, same
        "always allocated, not lazily" choice `pause()`/`resume()`'s events
        aren't (those genuinely can't exist before start() picks a thread).
        A message queued before the target ever calls `receive()` simply
        waits there -- no ordering requirement between send and receive.
        """
        if not isinstance(agent, SenseAgent):
            raise SenseRuntimeError(f"send() expects an Agent as its first argument, got {type_name(agent)}")
        agent._mailbox.put(message)

    def _receive_message(self, env: Environment, line: int, *args: Any) -> Any:
        """`receive(timeout?)`: the other half of `send()` -- blocks the
        calling agent's own thread until a message lands in its mailbox (or
        the optional timeout, in seconds, elapses, returning `nil`). Only
        legal inside a running agent's body, same `env.get_current_agent()`
        lookup `pause()` uses -- there is no "mailbox" for plain top-level
        code or a bare `fn` to receive into. Unlike `pause()`, this needs no
        hand-off protocol of its own (no `_resume_event`/`_settled_event`
        pair) -- `queue.Queue.get()` already blocks correctly on its own,
        so this is the first inter-thread primitive in the language that
        didn't need one.
        """
        if len(args) > 1:
            raise SenseRuntimeError("receive() expects 0 or 1 arguments (an optional timeout in seconds)", line)
        agent = env.get_current_agent()
        if agent is None:
            raise SenseRuntimeError("receive() can only be called inside a running agent's body", line)
        timeout = args[0] if args else None
        if timeout is not None and not isinstance(timeout, (int, float)):
            raise SenseRuntimeError(f"receive()'s timeout must be a number, got {type_name(timeout)}", line)
        try:
            return agent._mailbox.get(timeout=timeout)
        except queue.Empty:
            return None

    def _ask_human(self, env: Environment, line: int, question: Any) -> Any:
        """A human-intervention primitive, a builtin call rather than a
        `{ question = ... }` struct/record-literal form, since Sense has
        no struct/record-literal syntax yet. Built directly on
        pause()/resume(): inside a running agent, this *is* a pause carrying
        the question as the reason, and the resumer's answer becomes this
        call's return value. Outside any agent (a plain script, or the REPL),
        there's no one to hand control back to, so it falls back to reading a
        real line from stdin instead of suspending anything.
        """
        if not isinstance(question, str):
            raise SenseRuntimeError(f"ask_human() expects a String question, got {type_name(question)}", line)
        if env.get_current_agent() is not None:
            return self._pause_agent(env, line, question)
        self._emit(question)
        return self._read_line("> ")

    def _collect_agent_result(self, agent: SenseAgent) -> SenseAgent:
        if agent.status == "failed":
            error, agent._pending_error = agent._pending_error, None
            raise error
        return agent

    def _await(self, env: Environment, line: int, value: Any) -> Any:
        """`await(x)`: blocks until an `async agent`/`async tool` call
        settles, then returns its result (or re-raises its error). A no-op
        on anything already settled -- including every *sync* agent, which
        is always already settled by the time start()/resume() return -- so
        existing code never needs to change, and new code can `await()`
        uniformly regardless of sync/async."""
        if isinstance(value, SenseAgent):
            if value.status == "created":
                raise SenseRuntimeError(f"cannot await agent '{value.name}' — it hasn't been started yet", line)
            # Always wait on the event itself, not on .status: right after an
            # async resume() returns, .status can still briefly read "paused"
            # even though a resume is already in flight (the agent's own
            # thread hasn't transitioned it to "running" yet) -- waiting on
            # .status instead of the event would race and return stale state.
            # Event.wait() is a no-op if already settled, so this is correct
            # for a sync agent (already settled by the time start()/resume()
            # returned) too.
            value._settled_event.wait()
            return self._collect_agent_result(value)
        if isinstance(value, SenseFuture):
            value._done.wait()
            if value._error is not None:
                raise value._error
            return value._value
        raise SenseRuntimeError(f"await() expects an Agent or a Future, got {type_name(value)}", line)

    def _exec_ActionDecl(self, stmt: n.ActionDecl, env: Environment, file_path: str | None) -> None:
        action_def = SenseActionDef(decl=stmt, closure=env)
        env.define(stmt.name, action_def)
        self.declared_actions.append(action_def)

    def _exec_ToolDecl(self, stmt: n.ToolDecl, env: Environment, file_path: str | None) -> None:
        tool_def = SenseToolDef(decl=stmt, closure=env)
        env.define(stmt.name, tool_def)
        self.declared_tools.append(tool_def)

    @staticmethod
    def _default_memory_path(name: str, file_path: str | None) -> str:
        """Where a `persistent memory`'s SQLite file lands when no explicit
        path was given -- `persistent memory name: "path"` always wins
        over all of this; this only decides the *default*.

        Previously just `f"{name}.sense_memory.db"`, handed straight to
        `sqlite3.connect(...)` -- which resolves a relative path against
        the process's current working directory, not the .sns file's own
        directory. Two real symptoms of that: running the exact same
        script from two different terminals/directories silently used two
        different database files, and every default-location memory
        across an entire machine landed in one shared "wherever you
        happened to run `sense`" pile with no organization at all.

        Precedence, highest first:
          1. `SENSE_MEMORY_DIR` env var, if set -- an explicit, global,
             hassle-free override for anyone who wants every
             default-location memory in one place they chose (their own
             data directory, a synced folder, wherever). Flat under that
             directory (`<dir>/<name>.sense_memory.db`): setting this is
             an explicit opt-in to one shared location, so a same-named
             collision across two projects is a tradeoff the user chose,
             not an accident.
          2. The declaring `.sns` file's own directory, in a dedicated
             `.sense_memory/` subfolder -- consistent, predictable, and
             already the exact convention `import "./x.sns"` resolution
             uses (relative to the *file*, not the invoking shell), and
             scoped per-file so two unrelated projects both declaring
             `persistent memory ledger` can never collide.
          3. Falls back to the old CWD-relative behavior only when there's
             no real file to anchor to at all (e.g. the REPL, or
             `run_source` called directly with no `file_path`) -- there's
             no better default available in that case.

        A `@staticmethod` (no interpreter state needed) specifically so
        `sense memory-path <file>` (cli.py) can compute this without
        running the file at all -- a program's `persistent memory`
        declarations may have real, non-idempotent side effects the
        moment they run (creating a file, an initial schema), which a
        pure "where would this go" query shouldn't trigger.
        """
        filename = f"{name}.sense_memory.db"
        override_dir = os.environ.get("SENSE_MEMORY_DIR")
        if override_dir:
            return str(Path(override_dir) / filename)
        if file_path is not None:
            return str(Path(file_path).resolve().parent / ".sense_memory" / filename)
        return filename

    def _exec_MemoryDecl(self, stmt: n.MemoryDecl, env: Environment, file_path: str | None) -> None:
        if stmt.is_persistent:
            path = stmt.kind if stmt.kind is not None else self._default_memory_path(stmt.name, file_path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_log (
                    version   INTEGER PRIMARY KEY AUTOINCREMENT,
                    key       TEXT NOT NULL,
                    op        TEXT NOT NULL,
                    value     TEXT,
                    timestamp REAL NOT NULL
                )
                """
            )
            conn.commit()
            mem = SensePersistentMemory(name=stmt.name, path=path, _conn=conn)
            env.define(stmt.name, mem)
            self.declared_memories.append(mem)
            return
        mem = SenseMemory(name=stmt.name, kind=stmt.kind)
        env.define(stmt.name, mem)
        self.declared_memories.append(mem)

    # -- persistent memory: an append-only log, never UPDATE/DELETE -- that
    # one rule is what makes history()/as_of() possible with no separate
    # audit log or second code path. See values.py's SensePersistentMemory.

    def _pmem_check_key(self, key: Any, line: int) -> str:
        if not isinstance(key, str):
            raise SenseRuntimeError(f"memory key must be a String, got {type_name(key)}", line)
        return key

    def _pmem_remember(self, mem: SensePersistentMemory, key: Any, value: Any, line: int) -> None:
        key = self._pmem_check_key(key, line)
        try:
            encoded = json.dumps(value)
        except TypeError as exc:
            raise SenseRuntimeError(
                "persistent memory can only store JSON-serializable values "
                f"(Int/Float/String/Bool/Nil/Array of those): {type(exc).__name__}: {exc}",
                line,
            ) from exc
        with mem._lock:
            mem._conn.execute(
                "INSERT INTO memory_log (key, op, value, timestamp) VALUES (?, 'remember', ?, ?)",
                (key, encoded, time.time()),
            )
            mem._conn.commit()
        return None

    def _pmem_recall(self, mem: SensePersistentMemory, key: Any, line: int) -> Any:
        key = self._pmem_check_key(key, line)
        with mem._lock:
            row = mem._conn.execute(
                "SELECT op, value FROM memory_log WHERE key=? ORDER BY version DESC LIMIT 1", (key,)
            ).fetchone()
        if row is None or row[0] == "forget":
            return None
        return json.loads(row[1])

    def _pmem_forget(self, mem: SensePersistentMemory, key: Any, line: int) -> None:
        key = self._pmem_check_key(key, line)
        with mem._lock:
            mem._conn.execute(
                "INSERT INTO memory_log (key, op, value, timestamp) VALUES (?, 'forget', NULL, ?)",
                (key, time.time()),
            )
            mem._conn.commit()
        return None

    def _pmem_keys(self, mem: SensePersistentMemory) -> list[str]:
        with mem._lock:
            rows = mem._conn.execute(
                """
                SELECT key FROM memory_log m1
                WHERE op = 'remember'
                  AND version = (SELECT MAX(version) FROM memory_log m2 WHERE m2.key = m1.key)
                """
            ).fetchall()
        return [row[0] for row in rows]

    def _pmem_history(self, mem: SensePersistentMemory, key: Any, line: int) -> list[str]:
        key = self._pmem_check_key(key, line)
        with mem._lock:
            rows = mem._conn.execute(
                "SELECT version, op, value, timestamp FROM memory_log WHERE key=? ORDER BY version ASC", (key,)
            ).fetchall()
        lines: list[str] = []
        for version, op, value, ts in rows:
            when = time.ctime(ts)
            if op == "remember":
                lines.append(f"[v{version}] remember '{key}' = {stringify(json.loads(value))} ({when})")
            else:
                lines.append(f"[v{version}] forget '{key}' ({when})")
        return lines

    def _pmem_as_of(self, mem: SensePersistentMemory, key: Any, version: Any, line: int) -> Any:
        key = self._pmem_check_key(key, line)
        if not isinstance(version, int) or isinstance(version, bool):
            raise SenseRuntimeError(f"as_of() expects an Int version, got {type_name(version)}", line)
        with mem._lock:
            row = mem._conn.execute(
                "SELECT op, value FROM memory_log WHERE key=? AND version<=? ORDER BY version DESC LIMIT 1",
                (key, version),
            ).fetchone()
        if row is None or row[0] == "forget":
            return None
        return json.loads(row[1])

    def _exec_PolicyStmt(self, stmt: n.PolicyStmt, env: Environment, file_path: str | None) -> None:
        for rule in stmt.rules:
            env.set_policy_rule(rule.capability, rule.effect == "allow")

    def _audit(
        self, action: "SenseAction | SenseActionDef | SenseToolDef", event: str, detail: str | None,
        line: int | None, env: Environment,
    ) -> None:
        self.audit_log.append(
            AuditEntry(
                action_name=action.name, kind=action.kind, event=event, detail=detail, line=line,
                timestamp=time.time(), lineage=env.get_lineage(),
            )
        )

    def _prepare_action(self, action_def: SenseActionDef, args: list[Any], call_line: int, env: Environment) -> SenseAction:
        if len(args) != action_def.arity:
            raise SenseRuntimeError(
                f"'{action_def.name}' expects {action_def.arity} argument(s), got {len(args)}", call_line
            )
        bound: dict[str, Any] = {}
        for param, arg in zip(action_def.decl.params, args):
            if param.type_ann is not None and not matches_type(arg, param.type_ann):
                raise SenseTypeError(
                    f"argument '{param.name}' of '{action_def.name}' expects {param.type_ann}, got {type_name(arg)}",
                    call_line,
                )
            bound[param.name] = arg
        # Preparing an action must never run its body.
        action = SenseAction(action_def=action_def, args=bound, state="prepared")
        self._audit(action, "prepared", None, call_line, env)
        return action

    def _verify_action(self, env: Environment, line: int, action: Any) -> SenseAction:
        if not isinstance(action, SenseAction):
            raise SenseRuntimeError(f".verify() expects an Action, got {type_name(action)}", line)
        if action.state != "prepared":
            raise SenseRuntimeError(
                f"cannot verify action '{action.name}' — its state is '{action.state}', not 'prepared'", line
            )
        cap = action.action_def.decl.requires_capability
        if cap is not None and not env.get_capability(cap):
            action.state = "denied"
            self._audit(action, "denied", f"capability '{cap}' denied by policy", line, env)
            raise SensePolicyError(f"action '{action.name}' requires capability '{cap}', which is denied by policy", line)
        action.state = "verified"
        self._audit(action, "verified", None, line, env)
        return action

    def _approve_action(self, action: Any, line: int | None = None) -> SenseAction:
        if not isinstance(action, SenseAction):
            raise SenseRuntimeError(f".approve() expects an Action, got {type_name(action)}", line)
        if action.state not in ("prepared", "verified"):
            raise SenseRuntimeError(
                f"cannot approve action '{action.name}' — its state is '{action.state}', not 'prepared'/'verified'",
                line,
            )
        action.approved = True
        return action

    def _commit_action(self, action: SenseAction, env: Environment) -> Any:
        if action.state != "verified":
            raise SenseRuntimeError(
                f"action '{action.name}' must be verified before .commit() (current state: '{action.state}')"
            )
        # Re-check now, immediately before the external effect: a prior
        # .verify() must not be blindly trusted at commit time.
        cap = action.action_def.decl.requires_capability
        if cap is not None and not env.get_capability(cap):
            action.state = "denied"
            self._audit(action, "denied", f"capability '{cap}' denied by policy", None, env)
            raise SensePolicyError(f"action '{action.name}' requires capability '{cap}', which is denied by policy")
        if action.action_def.decl.requires_approval and not action.approved:
            action.state = "denied"
            self._audit(action, "denied", "requires approval", None, env)
            raise SenseApprovalError(
                f"action '{action.name}' requires approval before commit() — call {action.name}.approve() first"
            )

        # Dry run: every check above this point is real -- a denied
        # capability or missing approval still raises exactly as it would
        # in a normal run. Only the actual effect (the body) is skipped.
        # `state` stays "committed" (not a separate dry-run state) so every
        # other check in the language that keys off that exact string
        # (can this be rolled back, can commit() be called again) doesn't
        # need to learn a second spelling of "done" -- the audit log's
        # `detail` field is what marks this as simulated.
        if self.dry_run:
            action.state = "committed"
            self._audit(action, "committed", "dry run — body not executed", None, env)
            return None

        call_env = action.action_def.closure.child()
        call_env.call_frame = (action.kind, action.name)
        for name, value in action.args.items():
            call_env.define(name, value)
        # Saved before the body runs (not after) so a variable declared
        # partway through a body that then raises is still on the record --
        # harmless either way, since a "failed" action can never reach
        # .rollback() (state must be "committed"), but there's no reason
        # to make that guarantee do double duty here.
        action.commit_env = call_env
        try:
            self._exec_block_statements(action.action_def.decl.body.statements, call_env, None)
            result = None
        except _Return as ret:
            result = ret.value
        except Exception as exc:
            action.state = "failed"
            self._audit(action, "failed", str(exc), None, env)
            raise
        action.state = "committed"
        self._audit(action, "committed", None, None, env)
        return result

    def _rollback_action(self, env: Environment, line: int, action: Any) -> Any:
        if not isinstance(action, SenseAction):
            raise SenseRuntimeError(f".rollback() expects an Action, got {type_name(action)}", line)
        if action.kind != "reversible":
            raise SenseRuntimeError(
                f"cannot rollback '{action.name}' — only a reversible action can be rolled back", line
            )
        rollback_body = action.action_def.decl.rollback_body
        if rollback_body is None:
            # Not reachable via the parser any more -- `Parser._action_decl`
            # now requires a 'rollback' block on every 'reversible action',
            # so `kind == "reversible"` (just checked above) already implies
            # this is set. Kept as a real check anyway, not an assert: it's
            # the only thing standing between a malformed ActionDecl (built
            # by hand, not through the parser) and an AttributeError here.
            raise SenseRuntimeError(f"action '{action.name}' has no 'rollback' block to roll back with", line)
        if action.state != "committed":
            raise SenseRuntimeError(
                f"cannot rollback action '{action.name}' — its state is '{action.state}', not 'committed'", line
            )
        # Re-check now, same "never trust stale verification" principle
        # .commit() already applies — a rollback denied by policy leaves
        # the action's state at "committed": nothing changed, so nothing
        # should look changed.
        cap = action.action_def.decl.requires_capability
        if cap is not None and not env.get_capability(cap):
            self._audit(action, "denied", f"capability '{cap}' denied by policy (rollback)", line, env)
            raise SensePolicyError(f"action '{action.name}' requires capability '{cap}', which is denied by policy")

        # Same dry-run shape as _commit_action: the capability check above
        # is real, only the rollback body's effect is skipped.
        if self.dry_run:
            action.state = "rolled_back"
            self._audit(action, "rolled_back", "dry run — rollback not executed", line, env)
            return None

        # A child of the exact env .commit()'s body ran in, not a fresh
        # one off the closure -- so the rollback block can see whatever
        # the commit body computed (an inserted row's id, the value being
        # overwritten), the same way undoing something naturally needs to
        # reference what doing it produced. `commit_env` is always set by
        # this point in normal use: reaching here already required
        # state == "committed", which only a real (non-dry-run) .commit()
        # call sets -- checked anyway, not assumed, same reasoning the
        # rollback_body check above has: it's the only thing standing
        # between a hand-built SenseAction (state set without ever really
        # calling .commit()) and an AttributeError here.
        if action.commit_env is None:
            raise SenseRuntimeError(
                f"action '{action.name}' is 'committed' but was never actually run -- cannot roll back", line
            )
        call_env = action.commit_env.child()
        call_env.call_frame = ("rollback", action.name)
        try:
            self._exec_block_statements(rollback_body.statements, call_env, None)
            result = None
        except _Return as ret:
            result = ret.value
        except Exception as exc:
            # A failing rollback body leaves the action's state at
            # "committed" -- there's no safe recovery story for a partial
            # rollback failure here, so this is surfaced as an ordinary
            # error rather than pretending to auto-retry/undo it.
            self._audit(action, "failed", f"rollback body failed: {exc}", line, env)
            raise
        action.state = "rolled_back"
        self._audit(action, "rolled_back", None, line, env)
        return result

    def _exec_IfStmt(self, stmt: n.IfStmt, env: Environment, file_path: str | None) -> None:
        if is_truthy(self._evaluate(stmt.condition, env, file_path)):
            self._exec_Block(stmt.then_branch, env.child(), file_path)
        elif stmt.else_branch is not None:
            if isinstance(stmt.else_branch, n.IfStmt):
                self._execute(stmt.else_branch, env, file_path)
            else:
                self._exec_Block(stmt.else_branch, env.child(), file_path)

    def _exec_WhileStmt(self, stmt: n.WhileStmt, env: Environment, file_path: str | None) -> None:
        while is_truthy(self._evaluate(stmt.condition, env, file_path)):
            try:
                self._exec_Block(stmt.body, env.child(), file_path)
            except _Break:
                break
            except _Continue:
                continue

    def _exec_ForStmt(self, stmt: n.ForStmt, env: Environment, file_path: str | None) -> None:
        iterable = self._evaluate(stmt.iterable, env, file_path)
        if isinstance(iterable, str):
            items = list(iterable)
        elif isinstance(iterable, list):
            items = iterable
        else:
            raise SenseRuntimeError(f"cannot iterate over value of type {type_name(iterable)}", stmt.line)
        for item in items:
            loop_env = env.child()
            loop_env.define(stmt.var_name, item)
            try:
                self._exec_Block(stmt.body, loop_env, file_path)
            except _Break:
                break
            except _Continue:
                continue

    def _exec_ReturnStmt(self, stmt: n.ReturnStmt, env: Environment, file_path: str | None) -> None:
        value = self._evaluate(stmt.value, env, file_path) if stmt.value is not None else None
        raise _Return(value)

    def _exec_BreakStmt(self, stmt: n.BreakStmt, env: Environment, file_path: str | None) -> None:
        raise _Break()

    def _exec_ContinueStmt(self, stmt: n.ContinueStmt, env: Environment, file_path: str | None) -> None:
        raise _Continue()

    def _exec_ImportStmt(self, stmt: n.ImportStmt, env: Environment, file_path: str | None) -> None:
        if stmt.is_python:
            module = self._load_python_module(stmt.path, stmt.line)
            alias = stmt.alias or stmt.path.rsplit(".", 1)[-1]
            env.define(alias, module)
            return
        module = self._load_module(stmt.path, file_path, stmt.line)
        alias = stmt.alias or os.path.splitext(os.path.basename(stmt.path))[0]
        env.define(alias, module)

    def _exec_FromImportStmt(self, stmt: n.FromImportStmt, env: Environment, file_path: str | None) -> None:
        """`from X import a, b as c` -- loads the module exactly like
        `import` (same cache, so `import X` and `from X import ...` in the
        same program never load it twice), then binds each requested name
        directly instead of the whole module under one alias. A name the
        module doesn't actually have raises immediately -- the same
        SenseRuntimeError `<module>.name` member access already raises,
        not a new error shape."""
        if stmt.is_python:
            module = self._load_python_module(stmt.path, stmt.line)
            for name, alias in stmt.names:
                try:
                    attr = getattr(module.value, name)
                except AttributeError as exc:
                    raise SenseRuntimeError(f"{module.label} has no attribute '{name}'", stmt.line) from exc
                if callable(attr):
                    value: Any = BuiltinFunction(
                        name=f"<python>.{name}", arity=-1, fn=self._wrap_python_callable(attr, stmt.line)
                    )
                else:
                    value = to_sense_value(attr)
                env.define(alias or name, value)
            return
        module = self._load_module(stmt.path, file_path, stmt.line)
        for name, alias in stmt.names:
            if not module.env.has(name):
                raise SenseRuntimeError(f"module '{module.name}' has no member '{name}'", stmt.line)
            env.define(alias or name, module.env.get(name, stmt.line))

    def _load_python_module(self, module_name: str, line: int) -> PythonValue:
        """`import python "x.y"` reaches straight into an installed
        Python package. This grants that module *unrestricted* access —
        capability/policy gating only applies to calls a Sense `action`
        wraps, exactly the same boundary that already exists for a plain
        Sense `fn`; Python interop doesn't open a new hole, it extends the
        existing one to cover foreign calls too."""
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise SenseImportError(f"python module not found: {module_name!r} ({exc})", line) from exc
        return PythonValue(module, label=f"module '{module_name}'")

    def _exec_ExprStmt(self, stmt: n.ExprStmt, env: Environment, file_path: str | None) -> None:
        self._evaluate(stmt.expr, env, file_path)

    def _exec_Block(self, block: n.Block, env: Environment, file_path: str | None) -> None:
        self._exec_block_statements(block.statements, env, file_path)

    # -- module loading -------------------------------------------------------

    def _load_module(self, rel_path: str, importer_file: str | None, line: int) -> SenseModule:
        if importer_file is None:
            base_dir = os.getcwd()
        else:
            base_dir = os.path.dirname(importer_file)
        abs_path = os.path.normpath(os.path.join(base_dir, rel_path))

        if abs_path in self._module_cache:
            return self._module_cache[abs_path]
        if abs_path in self._modules_in_progress:
            raise SenseImportError(f"circular import detected: {abs_path}", line)
        if not os.path.isfile(abs_path):
            raise SenseImportError(f"module not found: {rel_path!r} (resolved to {abs_path})", line)

        self._modules_in_progress.add(abs_path)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                source = f.read()
            tokens = tokenize(source)
            program = parse(tokens)
            module_env = self.globals.child()
            self._exec_block_statements(program.statements, module_env, abs_path)
        finally:
            self._modules_in_progress.discard(abs_path)

        module_name = os.path.splitext(os.path.basename(abs_path))[0]
        module = SenseModule(name=module_name, env=module_env)
        self._module_cache[abs_path] = module
        return module

    # -- expression evaluation --------------------------------------------------

    def _evaluate(self, expr: n.Node, env: Environment, file_path: str | None) -> Any:
        method = getattr(self, f"_eval_{type(expr).__name__}", None)
        if method is None:
            raise SenseRuntimeError(f"no evaluator for expression {type(expr).__name__}", expr.line)
        return method(expr, env, file_path)

    def _eval_IntLit(self, expr: n.IntLit, env, file_path):
        return expr.value

    def _eval_FloatLit(self, expr: n.FloatLit, env, file_path):
        return expr.value

    def _eval_StringLit(self, expr: n.StringLit, env, file_path):
        return expr.value

    def _eval_BoolLit(self, expr: n.BoolLit, env, file_path):
        return expr.value

    def _eval_NilLit(self, expr: n.NilLit, env, file_path):
        return None

    def _eval_ArrayLit(self, expr: n.ArrayLit, env, file_path):
        return [self._evaluate(el, env, file_path) for el in expr.elements]

    def _eval_Identifier(self, expr: n.Identifier, env: Environment, file_path):
        return env.get(expr.name, expr.line)

    def _eval_Assign(self, expr: n.Assign, env: Environment, file_path):
        value = self._evaluate(expr.value, env, file_path)
        env.define_or_assign(expr.name, value)
        return value

    def _eval_IndexAssign(self, expr: n.IndexAssign, env: Environment, file_path):
        target = self._evaluate(expr.target, env, file_path)
        index = self._evaluate(expr.index, env, file_path)
        value = self._evaluate(expr.value, env, file_path)
        if isinstance(target, PythonValue):
            try:
                target.value[from_sense_value(index)] = from_sense_value(value)
            except Exception as exc:  # noqa: BLE001 - surface whatever the wrapped Python object raised
                raise SenseRuntimeError(
                    f"python error setting index: {type(exc).__name__}: {exc}", expr.line
                ) from exc
            return value
        if not isinstance(target, list):
            raise SenseRuntimeError(f"cannot index-assign into type {type_name(target)}", expr.line)
        if not isinstance(index, int) or isinstance(index, bool):
            raise SenseRuntimeError(f"array index must be Int, got {type_name(index)}", expr.line)
        try:
            target[index] = value
        except IndexError as exc:
            raise SenseRuntimeError(f"array index {index} out of bounds", expr.line) from exc
        return value

    def _eval_UnaryOp(self, expr: n.UnaryOp, env: Environment, file_path):
        operand = self._evaluate(expr.operand, env, file_path)
        if expr.op == "-":
            if not isinstance(operand, (int, float)) or isinstance(operand, bool):
                raise SenseRuntimeError(f"unary '-' requires a number, got {type_name(operand)}", expr.line)
            return -operand
        if expr.op == "not":
            return not is_truthy(operand)
        raise SenseRuntimeError(f"unknown unary operator {expr.op!r}", expr.line)

    def _eval_LogicalOp(self, expr: n.LogicalOp, env: Environment, file_path):
        left = self._evaluate(expr.left, env, file_path)
        if expr.op == "or":
            if is_truthy(left):
                return left
            return self._evaluate(expr.right, env, file_path)
        else:  # "and"
            if not is_truthy(left):
                return left
            return self._evaluate(expr.right, env, file_path)

    def _eval_BinaryOp(self, expr: n.BinaryOp, env: Environment, file_path):
        left = self._evaluate(expr.left, env, file_path)
        right = self._evaluate(expr.right, env, file_path)
        op = expr.op

        if op == "+":
            if isinstance(left, str) or isinstance(right, str):
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                raise SenseRuntimeError(
                    f"'+' requires both operands to be String (got {type_name(left)} and {type_name(right)}); "
                    "use str() to convert",
                    expr.line,
                )
            if isinstance(left, list) and isinstance(right, list):
                return left + right
            self._require_numeric(left, right, "+", expr.line)
            return left + right
        if op == "-":
            self._require_numeric(left, right, "-", expr.line)
            return left - right
        if op == "*":
            self._require_numeric(left, right, "*", expr.line)
            return left * right
        if op == "/":
            self._require_numeric(left, right, "/", expr.line)
            if right == 0:
                raise SenseRuntimeError("division by zero", expr.line)
            return left / right  # '/' is always true (Float) division, by design
        if op == "%":
            self._require_numeric(left, right, "%", expr.line)
            if right == 0:
                raise SenseRuntimeError("modulo by zero", expr.line)
            return left % right
        if op == "==":
            return self._values_equal(left, right)
        if op == "!=":
            return not self._values_equal(left, right)
        if op in ("<", "<=", ">", ">="):
            both_strings = isinstance(left, str) and isinstance(right, str)
            if not both_strings:
                self._require_numeric(left, right, op, expr.line)
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == ">":
                return left > right
            return left >= right

        raise SenseRuntimeError(f"unknown binary operator {op!r}", expr.line)

    def _require_numeric(self, left: Any, right: Any, op: str, line: int) -> None:
        def is_num(v: Any) -> bool:
            return isinstance(v, (int, float)) and not isinstance(v, bool)

        if not (is_num(left) and is_num(right)):
            raise SenseRuntimeError(
                f"'{op}' requires numeric operands, got {type_name(left)} and {type_name(right)}", line
            )

    def _values_equal(self, left: Any, right: Any) -> bool:
        if isinstance(left, bool) != isinstance(right, bool):
            return False
        return left == right

    def _eval_Index(self, expr: n.Index, env: Environment, file_path):
        target = self._evaluate(expr.target, env, file_path)
        index = self._evaluate(expr.index, env, file_path)
        if isinstance(target, (list, str)):
            if not isinstance(index, int) or isinstance(index, bool):
                raise SenseRuntimeError(f"index must be Int, got {type_name(index)}", expr.line)
            try:
                return target[index]
            except IndexError as exc:
                raise SenseRuntimeError(f"index {index} out of bounds", expr.line) from exc
        if isinstance(target, PythonValue):
            try:
                return to_sense_value(target.value[from_sense_value(index)])
            except Exception as exc:  # noqa: BLE001 - surface whatever the wrapped Python object raised
                raise SenseRuntimeError(
                    f"python error indexing: {type(exc).__name__}: {exc}", expr.line
                ) from exc
        raise SenseRuntimeError(f"cannot index into type {type_name(target)}", expr.line)

    def _eval_MemberAccess(self, expr: n.MemberAccess, env: Environment, file_path):
        target = self._evaluate(expr.target, env, file_path)
        if isinstance(target, SenseModule):
            if not target.env.has(expr.name):
                raise SenseRuntimeError(f"module '{target.name}' has no member '{expr.name}'", expr.line)
            return target.env.get(expr.name, expr.line)
        if isinstance(target, SenseModel):
            if expr.name == "name":
                return target.name
            if expr.name == "ask":
                return BuiltinFunction(
                    name=f"{target.name}.ask",
                    arity=1,
                    fn=lambda prompt, _m=target, _e=env, _l=expr.line: self._ask_model(_m, prompt, _e, _l),
                )
            raise SenseRuntimeError(f"model '{target.name}' has no member '{expr.name}'", expr.line)
        if isinstance(target, Answer):
            if expr.name in ("value", "confidence", "source"):
                return getattr(target, expr.name)
            raise SenseRuntimeError(f"answer has no member '{expr.name}'", expr.line)
        if isinstance(target, SenseAgent):
            if expr.name in ("status", "name", "pause_reason", "doc"):
                return getattr(target, expr.name)
            if target.env.has(expr.name):
                return target.env.get(expr.name, expr.line)
            raise SenseRuntimeError(f"agent '{target.name}' has no member '{expr.name}'", expr.line)
        if isinstance(target, SenseAction):
            if expr.name == "commit":
                return BuiltinFunction(
                    name=f"{target.name}.commit",
                    arity=0,
                    fn=lambda _a=target, _e=env: self._commit_action(_a, _e),
                )
            if expr.name == "verify":
                return BuiltinFunction(
                    name=f"{target.name}.verify",
                    arity=0,
                    fn=lambda _a=target, _e=env, _l=expr.line: self._verify_action(_e, _l, _a),
                )
            if expr.name == "approve":
                return BuiltinFunction(
                    name=f"{target.name}.approve",
                    arity=0,
                    fn=lambda _a=target, _l=expr.line: self._approve_action(_a, _l),
                )
            if expr.name == "rollback":
                return BuiltinFunction(
                    name=f"{target.name}.rollback",
                    arity=0,
                    fn=lambda _a=target, _e=env, _l=expr.line: self._rollback_action(_e, _l, _a),
                )
            if expr.name in ("state", "name", "kind", "approved"):
                return getattr(target, expr.name)
            raise SenseRuntimeError(f"action '{target.name}' has no member '{expr.name}'", expr.line)
        if isinstance(target, SenseMemory):
            def _check_key(key: Any) -> str:
                if not isinstance(key, str):
                    raise SenseRuntimeError(f"memory key must be a String, got {type_name(key)}", expr.line)
                return key

            if expr.name == "remember":
                def _remember(key: Any, value: Any, _m=target) -> None:
                    _m.store[_check_key(key)] = value
                    return None

                return BuiltinFunction(name=f"{target.name}.remember", arity=2, fn=_remember)
            if expr.name == "recall":
                return BuiltinFunction(
                    name=f"{target.name}.recall", arity=1, fn=lambda key, _m=target: _m.store.get(_check_key(key))
                )
            if expr.name == "forget":
                def _forget(key: Any, _m=target) -> None:
                    _m.store.pop(_check_key(key), None)
                    return None

                return BuiltinFunction(name=f"{target.name}.forget", arity=1, fn=_forget)
            if expr.name == "keys":
                return BuiltinFunction(name=f"{target.name}.keys", arity=0, fn=lambda _m=target: list(_m.store.keys()))
            if expr.name in ("name", "kind"):
                return getattr(target, expr.name)
            raise SenseRuntimeError(f"memory '{target.name}' has no member '{expr.name}'", expr.line)
        if isinstance(target, SensePersistentMemory):
            if expr.name == "remember":
                return BuiltinFunction(
                    name=f"{target.name}.remember",
                    arity=2,
                    fn=lambda key, value, _m=target: self._pmem_remember(_m, key, value, expr.line),
                )
            if expr.name == "recall":
                return BuiltinFunction(
                    name=f"{target.name}.recall", arity=1, fn=lambda key, _m=target: self._pmem_recall(_m, key, expr.line)
                )
            if expr.name == "forget":
                return BuiltinFunction(
                    name=f"{target.name}.forget", arity=1, fn=lambda key, _m=target: self._pmem_forget(_m, key, expr.line)
                )
            if expr.name == "keys":
                return BuiltinFunction(name=f"{target.name}.keys", arity=0, fn=lambda _m=target: self._pmem_keys(_m))
            if expr.name == "history":
                return BuiltinFunction(
                    name=f"{target.name}.history", arity=1, fn=lambda key, _m=target: self._pmem_history(_m, key, expr.line)
                )
            if expr.name == "as_of":
                return BuiltinFunction(
                    name=f"{target.name}.as_of",
                    arity=2,
                    fn=lambda key, version, _m=target: self._pmem_as_of(_m, key, version, expr.line),
                )
            if expr.name in ("name", "path"):
                return getattr(target, expr.name)
            raise SenseRuntimeError(f"memory '{target.name}' has no member '{expr.name}'", expr.line)
        if isinstance(target, SenseSkill):
            if expr.name in ("name", "description", "tools"):
                return getattr(target, expr.name)
            raise SenseRuntimeError(f"skill '{target.name}' has no member '{expr.name}'", expr.line)
        if isinstance(target, SenseMcpServer):
            if expr.name in ("name", "tools"):
                return getattr(target, expr.name)
            if expr.name == "close":
                return BuiltinFunction(name=f"{target.name}.close", arity=0, fn=lambda _s=target: _s._bridge.close())
            raise SenseRuntimeError(f"mcp server '{target.name}' has no member '{expr.name}'", expr.line)
        if isinstance(target, SenseFuture):
            if expr.name == "done":
                return target.done  # a cheap non-blocking poll -- await() is still needed for the value
            raise SenseRuntimeError(f"future '{target.label}' has no member '{expr.name}'", expr.line)
        if isinstance(target, PythonValue):
            try:
                attr = getattr(target.value, expr.name)
            except AttributeError as exc:
                raise SenseRuntimeError(f"{target.label} has no attribute '{expr.name}'", expr.line) from exc
            if callable(attr):
                return BuiltinFunction(name=f"<python>.{expr.name}", arity=-1, fn=self._wrap_python_callable(attr, expr.line))
            return to_sense_value(attr)
        raise SenseRuntimeError(f"cannot access member '{expr.name}' on type {type_name(target)}", expr.line)

    def _wrap_python_callable(self, py_callable, line: int):
        def call(*args: Any) -> Any:
            py_args = [from_sense_value(a) for a in args]
            try:
                result = py_callable(*py_args)
            except Exception as exc:  # noqa: BLE001 - a foreign call can raise literally anything
                name = getattr(py_callable, "__name__", "python callable")
                raise SenseRuntimeError(
                    f"python error calling {name}: {type(exc).__name__}: {exc}", line
                ) from exc
            return to_sense_value(result)

        return call

    def _eval_Call(self, expr: n.Call, env: Environment, file_path):
        callee = self._evaluate(expr.callee, env, file_path)
        args = [self._evaluate(a, env, file_path) for a in expr.args]
        kwargs = {label: self._evaluate(v, env, file_path) for label, v in expr.kwargs}

        if isinstance(callee, BuiltinFunction):
            if not callee.accepts_kwargs and kwargs:
                raise SenseRuntimeError(
                    f"'{callee.name}' does not accept labeled arguments "
                    f"({', '.join(kwargs)})",
                    expr.line,
                )
            if callee.arity >= 0 and len(args) != callee.arity:
                raise SenseRuntimeError(
                    f"'{callee.name}' expects {callee.arity} argument(s), got {len(args)}", expr.line
                )
            if callee.needs_context:
                return callee.fn(env, expr.line, *args, **kwargs)
            return callee.fn(*args, **kwargs)

        if kwargs:
            raise SenseRuntimeError(
                f"labeled arguments are only supported for a handful of builtins "
                f"(e.g. inference(...)) -- not {type_name(callee)}",
                expr.line,
            )

        if isinstance(callee, SenseFunction):
            return self._call_function(callee, args, expr.line)

        if isinstance(callee, SenseActionDef):
            return self._prepare_action(callee, args, expr.line, env)

        if isinstance(callee, SenseToolDef):
            return self._call_tool(callee, args, env, expr.line)

        raise SenseRuntimeError(f"value of type {type_name(callee)} is not callable", expr.line)

    def _call_function(self, fn: SenseFunction, args: list[Any], call_line: int) -> Any:
        if len(args) != fn.arity:
            raise SenseRuntimeError(
                f"'{fn.name}' expects {fn.arity} argument(s), got {len(args)}", call_line
            )
        call_env = fn.closure.child()
        for param, arg in zip(fn.decl.params, args):
            if param.type_ann is not None and not matches_type(arg, param.type_ann):
                raise SenseTypeError(
                    f"argument '{param.name}' of '{fn.name}' expects {param.type_ann}, got {type_name(arg)}",
                    call_line,
                )
            call_env.define(param.name, arg)

        try:
            self._exec_Block(fn.decl.body, call_env, None)
            result = None
        except _Return as ret:
            result = ret.value

        if fn.decl.return_type is not None and not matches_type(result, fn.decl.return_type):
            raise SenseTypeError(
                f"'{fn.name}' declared to return {fn.decl.return_type}, got {type_name(result)}",
                call_line,
            )
        return result

    def _call_tool(self, tool: SenseToolDef, args: list[Any], env: Environment, call_line: int) -> Any:
        if len(args) != tool.arity:
            raise SenseRuntimeError(
                f"'{tool.name}' expects {tool.arity} argument(s), got {len(args)}", call_line
            )
        cap = tool.decl.requires_capability
        if cap is not None and not env.get_capability(cap):
            self._audit(tool, "denied", f"capability '{cap}' denied by policy", call_line, env)
            raise SensePolicyError(
                f"tool '{tool.name}' requires capability '{cap}', which is denied by policy", call_line
            )
        call_env = tool.closure.child()
        call_env.call_frame = ("tool", tool.name)
        for param, arg in zip(tool.decl.params, args):
            if param.type_ann is not None and not matches_type(arg, param.type_ann):
                raise SenseTypeError(
                    f"argument '{param.name}' of '{tool.name}' expects {param.type_ann}, got {type_name(arg)}",
                    call_line,
                )
            call_env.define(param.name, arg)
        self._audit(tool, "called", None, call_line, env)

        if tool.decl.is_async:
            return self._call_tool_async(tool, call_env, call_line)

        try:
            self._exec_Block(tool.decl.body, call_env, None)
            result = None
        except _Return as ret:
            result = ret.value

        if tool.decl.return_type is not None and not matches_type(result, tool.decl.return_type):
            raise SenseTypeError(
                f"'{tool.name}' declared to return {tool.decl.return_type}, got {type_name(result)}",
                call_line,
            )
        return result

    def _call_mcp_tool(self, tool: McpTool, arguments: dict[str, Any], env: Environment, line: int | None) -> str:
        """A parallel path to `_call_tool`, not through it -- an MCP tool
        has no `ToolDecl`/Sense body to execute, only a remote name and
        JSON schema. Same shape as `_call_tool` anyway: capability check
        against the caller's policy scope, audit entry, wrapped errors --
        a model's request to an external tool gets no more trust than a
        request to a local one does."""
        cap = tool.capability
        if cap is not None and not env.get_capability(cap):
            self.audit_log.append(
                AuditEntry(
                    action_name=tool.qualified_name, kind=tool.kind,
                    event="denied", detail=f"capability '{cap}' denied by policy", line=line, timestamp=time.time(),
                    lineage=env.get_lineage(),
                )
            )
            raise SensePolicyError(
                f"mcp tool '{tool.qualified_name}' requires capability '{cap}', which is denied by policy", line
            )
        try:
            result_text = tool.server._bridge.call_tool(tool.name, arguments)
        except Exception as exc:
            self.audit_log.append(
                AuditEntry(
                    action_name=tool.qualified_name, kind=tool.kind,
                    event="failed", detail=f"{type(exc).__name__}: {exc}", line=line, timestamp=time.time(),
                    lineage=env.get_lineage(),
                )
            )
            raise SenseRuntimeError(
                f"mcp tool '{tool.qualified_name}' failed: {type(exc).__name__}: {exc}", line
            ) from exc
        self.audit_log.append(
            AuditEntry(
                action_name=tool.qualified_name, kind=tool.kind,
                event="called", detail=None, line=line, timestamp=time.time(), lineage=env.get_lineage(),
            )
        )
        return result_text

    def _call_tool_async(self, tool: SenseToolDef, call_env: Environment, call_line: int) -> SenseFuture:
        """Runs an `async tool`'s body on its own daemon thread and returns
        a `SenseFuture` immediately -- capability/arity/argument checks
        already happened synchronously in `_call_tool`, so only the body
        itself (and its return-type check, since the value doesn't exist
        until the body finishes) happens on the background thread."""
        future = SenseFuture(label=f"tool '{tool.name}'")

        def run() -> None:
            try:
                try:
                    self._exec_Block(tool.decl.body, call_env, None)
                    result = None
                except _Return as ret:
                    result = ret.value
                if tool.decl.return_type is not None and not matches_type(result, tool.decl.return_type):
                    raise SenseTypeError(
                        f"'{tool.name}' declared to return {tool.decl.return_type}, got {type_name(result)}",
                        call_line,
                    )
                future._value = result
            except Exception as exc:  # noqa: BLE001 - handed to await(), not lost
                future._error = exc
            finally:
                future._done.set()

        threading.Thread(target=run, daemon=True).start()
        return future

