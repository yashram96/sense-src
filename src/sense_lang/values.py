"""Runtime value representation and type-annotation checking.

Sense values map onto native Python values for the deterministic core:

    Int    -> python int (excluding bool)
    Float  -> python float
    String -> python str
    Bool   -> python bool
    Nil    -> python None
    Array  -> python list
    Function / Builtin / Module -> dedicated wrapper classes below
"""

from __future__ import annotations

import inspect
import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from . import ast_nodes as n
from .providers import ModelProvider


def _extract_doc(body: "n.Block") -> str | None:
    """Python's own docstring convention, made to actually work: a bare
    string as the *first* statement of a `tool`/`action`/`agent`/plain
    function's body is captured as that declaration's `.doc`, not just an
    inert expression statement a human happens to read in source. No new
    keyword -- the string was already legal to write there (an ordinary
    `ExprStmt` wrapping a `StringLit`, ignored like any expression whose
    value goes unused); this only decides what it *means*. Works with an
    ordinary `"..."` (which already tolerates an embedded literal newline)
    or the newer `\"\"\"...\"\"\"` form the lexer now also accepts -- both
    produce the identical `StringLit` node, so there's nothing to
    distinguish here. `nil` (no doc) if the body is empty or doesn't start
    with a bare string."""
    if not body.statements:
        return None
    first = body.statements[0]
    if isinstance(first, n.ExprStmt) and isinstance(first.expr, n.StringLit):
        # A multi-line docstring's literal content carries the source's own
        # indentation baked in (every line after the first is indented to
        # match the body) plus the blank line right after the opening
        # `\"\"\"`/`"` -- the exact same shape Python's own docstrings have,
        # so `inspect.cleandoc` (strip common leading whitespace, drop
        # leading/trailing blank lines) is the right, already-standard fix
        # rather than writing a bespoke dedent.
        return inspect.cleandoc(first.expr.value)
    return None


@dataclass
class SenseFunction:
    decl: n.FnDecl
    closure: "Environment"

    @property
    def name(self) -> str:
        return self.decl.name

    @property
    def arity(self) -> int:
        return len(self.decl.params)

    @property
    def doc(self) -> str | None:
        return _extract_doc(self.decl.body)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<fn {self.name}>"


@dataclass
class BuiltinFunction:
    name: str
    arity: int  # -1 means variadic
    fn: Callable[..., Any]
    needs_context: bool = False  # if True, called as fn(env, line, *args)
    # If True, called as fn(..., **kwargs) too -- a labeled call argument
    # (`inference(..., temperature: 0.2)`) is only legal against a builtin
    # that opts in; every other builtin (and every fn/tool/action) raises
    # if one is given. See interpreter.py's _eval_Call and
    # ast_nodes.py's Call.kwargs.
    accepts_kwargs: bool = False

    def __repr__(self) -> str:  # pragma: no cover
        return f"<builtin {self.name}>"


@dataclass
class SenseModule:
    name: str
    env: "Environment"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<module {self.name}>"


@dataclass
class SenseModel:
    """A runtime reasoning resource, not a raw API call -- programs only
    ever see this wrapper, never a raw API handle.
    """

    name: str
    provider: ModelProvider

    def __repr__(self) -> str:  # pragma: no cover
        return f"<model {self.name}>"


@dataclass
class Answer:
    """The result of `ask(...)`: a value the runtime is not certain of.

    This type was originally named `Belief`; renamed because the word
    read as philosophical/vague for what it actually is: programs must
    be able to say "I infer this with
    N% confidence" instead of treating AI output as a plain, trusted
    value. `.value` access is always explicit — there is no implicit
    unwrapping.
    """

    value: Any
    confidence: float | None
    source: str  # the Model name that produced this answer

    def __repr__(self) -> str:  # pragma: no cover
        return f"<answer {self.value!r} confidence={self.confidence}>"


@dataclass
class SenseAgent:
    """A runtime entity, not just an executed block: it pursues, where a
    session merely bounds execution. Declaring `agent x: ...`
    creates this in "created" status without running its body; the builtin
    `start(agent)` runs it once, advancing status to "running" then
    "paused" (via `pause()` inside the body), "completed", or "failed".
    `env` is the agent's own persistent scope — state set during its run is
    inspectable afterward via member access (`agent_value.some_var`), same
    mechanism as `SenseModule`.

    Pause/resume runs the body on its own OS thread, cooperatively handed
    off via two `threading.Event`s (`_settled_event`, `_resume_event`) so
    exactly one of {the caller, the agent} is ever actually executing at a
    time — see `Interpreter._start_agent`/`_pause_agent`/`_resume_agent`.
    Not covered here: real concurrency/scheduling, or serialization
    across process restarts.

    `_mailbox` is a thread-safe FIFO queue (Python's `queue.Queue`, not a
    new synchronization primitive of Sense's own) allocated the moment the
    agent is declared -- not at `start()` time -- so `send(agent, msg)`
    works even before the agent has been started, or from any number of
    concurrent senders (e.g. several `async agent`s all messaging one
    coordinator) without a race, the same "reuse a stdlib primitive rather
    than hand-roll one" call `SensePersistentMemory._lock` made earlier.
    `receive(timeout?)` (inside the agent's own body only, same
    `env.get_current_agent()` lookup `pause()` uses) blocks on it directly
    -- see `Interpreter._send_message`/`_receive_message`.

    Still not built: capability/policy enforcement on the body's other
    calls, budget, multi-agent scheduling, agent spawning another agent's
    *declaration* from inside a body (only already-declared agents can be
    started/messaged).
    """

    name: str
    decl: "n.AgentDecl"
    env: "Environment"
    status: str = "created"
    pause_reason: str | None = None
    _thread: Any = None
    _resume_event: Any = None
    _settled_event: Any = None
    _pending_error: Any = None
    _resume_value: Any = None
    _mailbox: Any = field(default_factory=queue.Queue)

    @property
    def doc(self) -> str | None:
        return _extract_doc(self.decl.body)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<agent {self.name} status={self.status}>"


@dataclass
class SenseActionDef:
    """The declaration of a `reversible`/`irreversible action` — callable,
    like `SenseFunction`, but calling it does not run the body. It produces
    a `SenseAction` in "prepared" state instead -- the central law:
    preparing an action must not cause its external effect."""

    decl: "n.ActionDecl"
    closure: "Environment"

    @property
    def name(self) -> str:
        return self.decl.name

    @property
    def kind(self) -> str:
        return self.decl.kind

    @property
    def arity(self) -> int:
        return len(self.decl.params)

    @property
    def doc(self) -> str | None:
        return _extract_doc(self.decl.body)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.kind} action {self.name}>"


@dataclass
class SenseAction:
    """One prepared call of an `action` — not yet executed.

    Lifecycle: prepared -> verified -> committed (or denied/failed at
    either check). `<action>.verify()` and `<action>.commit()` each
    independently re-check the required capability against policy --
    verification is not a permanent guarantee, commit must not blindly
    trust an earlier verify.

    Every lifecycle step (`.verify()`, `.approve()`, `.commit()`,
    `.rollback()`) is a method on the action itself, never a free function
    taking the action as an argument -- one consistent shape, not a mix.

    `approved` is only meaningful when the action's declaration includes
    `requires approval` — set it via the `<action>.approve()` method. It's
    tracked on the action instance (not the definition) since each prepared
    call needs its own independent approval, the same way each call gets
    its own `args`.
    """

    action_def: SenseActionDef
    args: dict[str, Any]
    state: str = "prepared"
    approved: bool = False
    # The exact environment `.commit()`'s body ran in (params bound, plus
    # whatever local variables the body itself declared) -- kept so
    # `.rollback()` can see them too: `rollback:` is conceptually "undo
    # what commit just did," which routinely needs a value commit
    # computed (an inserted row's id, the value being overwritten), not
    # just the action's own original args. `.rollback()`'s own env is a
    # child of this one, not this one directly, so it gets its own scope
    # layer the same way any other nested block does. `None` until
    # `.commit()` actually runs a real (non-dry-run) body.
    commit_env: "Environment | None" = None

    @property
    def name(self) -> str:
        return self.action_def.name

    @property
    def kind(self) -> str:
        return self.action_def.kind

    def __repr__(self) -> str:  # pragma: no cover
        return f"<action {self.name} ({self.kind}) state={self.state}>"


@dataclass
class SenseToolDef:
    """A `tool` declaration — callable like `SenseFunction`, but every call
    is capability-checked (if `requires` was given) against the *caller's*
    policy scope before the body runs. Unlike `SenseActionDef`, calling one
    runs the body immediately: no prepare/verify/commit lifecycle. `kind`
    exists only so `Interpreter._audit` (built for actions) can log tool
    calls too without a separate code path."""

    decl: "n.ToolDecl"
    closure: "Environment"

    @property
    def name(self) -> str:
        return self.decl.name

    @property
    def kind(self) -> str:
        return "tool"

    @property
    def arity(self) -> int:
        return len(self.decl.params)

    @property
    def description(self) -> str | None:
        return self.decl.description

    @property
    def doc(self) -> str | None:
        return _extract_doc(self.decl.body)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<tool {self.name}>"


@dataclass
class SenseMemory:
    """A `memory` declaration's runtime value: a persistent-for-the-run
    key/value store, distinct from an ordinary variable. `kind` is a
    free-form, unvalidated hint. `store` is an in-process dict —
    explicitly not durable across runs, same "in-memory only" honesty as
    `Interpreter.audit_log`."""

    name: str
    kind: str | None = None
    store: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<memory {self.name} ({len(self.store)} entries)>"


@dataclass
class SensePersistentMemory:
    """A `persistent memory` declaration's runtime value: a durable,
    versioned, append-only key/value store backed by a SQLite file
    (`_conn`, opened by the interpreter, not here -- mirrors how
    `SenseAgent`'s `_thread`/`_resume_event` are runtime-only fields).

    Every `remember`/`forget` is a new row, never an UPDATE/DELETE -- that
    one rule is what makes `history(key)`/`as_of(key, version)` possible
    without a separate audit log or a second code path. Deliberately not
    Delta Lake (a distributed engine over Parquet/object storage -- a scale
    mismatch here): SQLite is stdlib and gives real ACID transactions at
    the file level.

    `_lock` guards `_conn`: a single `sqlite3.Connection` object is not
    safe for two threads to call `.execute()` on at the same instant even
    with `check_same_thread=False` (Python's sqlite3 module raises
    "bad parameter or other API misuse" under that race, found while
    testing `async agent`/`async tool` writing to the same persistent
    memory concurrently) -- this lock is what actually makes concurrent
    access safe, not SQLite alone."""

    name: str
    path: str
    _conn: Any = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<memory {self.name} (persistent, {self.path})>"


@dataclass
class SenseSkill:
    """A named, reusable bundle of `tool`s with its own description --
    no dedicated keyword: the `skill(...)` builtin already returns this
    runtime type, so the meaning is carried wherever it's checked (here,
    `ask_with_tools`), not by new grammar.

    `description` is not just documentation -- `Interpreter._ask_with_tools`
    prepends it (alongside every other skill's) to the prompt as an
    "Available skills:" preamble when this skill is included in an
    `ask_with_tools` call, so it has real, not cosmetic, effect."""

    name: str
    description: str
    tools: list["SenseToolDef"]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<skill {self.name} ({len(self.tools)} tools)>"


@dataclass
class McpTool:
    """One remote tool discovered from a connected MCP server -- no
    Sense body at all; invoking one is a JSON-RPC call to an external
    process, bridged via `mcp_client.McpBridge`. `type_of()` reports this
    as `"Tool"`, not a separate type: from a Sense program's perspective
    (and `ask_with_tools`'s own validation) it fills the exact same
    role a `SenseToolDef` does -- only `Interpreter._ask_with_tools`'s
    dispatch needs to know the difference, to call
    `Interpreter._call_mcp_tool` instead of `_call_tool`. Only ever
    meaningful as an element of `ask_with_tools`'s `tools` array or a
    `skill(...)` bundle -- there is no Sense identifier or call syntax
    for one, since Sense has no dynamic-dispatch call for "whatever this
    runtime string names."""

    server: "SenseMcpServer"
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def capability(self) -> str | None:
        return self.server.capability

    @property
    def kind(self) -> str:
        return "tool"  # so Interpreter._audit (built for actions/tools) can log MCP calls too, same as SenseToolDef

    @property
    def qualified_name(self) -> str:
        """`server.tool` -- what the model sees and echoes back
        (`ToolSpec.name`) and what audit entries name, so two servers (or
        a server and a bare Sense tool) can't collide on a generic name
        like "search". `.name` itself stays bare, since that's what the
        remote server's own `call_tool(name, ...)` actually expects."""
        return f"{self.server.name}.{self.name}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<mcp tool {self.qualified_name}>"


@dataclass
class SenseMcpServer:
    """A connection to an external MCP server, spawned as a subprocess
    over stdio and held open for as long as this value is used --
    `_bridge` (an `mcp_client.McpBridge`) owns the background thread and
    persistent event loop. `capability`, if given, gates every tool on
    this connection, checked at each individual call (not once at
    connection time) -- the same "never trust a stale check" principle
    `commit()`/`_call_tool` already apply. Unlike agent threads or SQLite
    connections elsewhere in this codebase, `close()` is an explicit
    member, not automatic: a connection spawns a real OS subprocess, a
    more visible failure mode (an orphaned process) than an idle thread
    or open file handle if left uncleaned."""

    name: str
    capability: str | None
    tools: list[McpTool]
    _bridge: Any = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<mcp server {self.name} ({len(self.tools)} tools)>"


@dataclass
class SenseFuture:
    """The result of calling an `async tool` — a pending value, not yet
    known. The body runs on its own daemon thread; `await(future)` blocks
    until it's done, then returns `_value` or re-raises `_error`. There is
    no cancellation, no timeout, and no thread pooling."""

    label: str  # for stringify()/error messages, e.g. "tool 'search'"
    _done: threading.Event = field(default_factory=threading.Event)
    _value: Any = None
    _error: BaseException | None = None

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def __repr__(self) -> str:  # pragma: no cover
        return f"<future {self.label}>"


@dataclass
class PythonValue:
    """A Python value Sense's native types can't represent directly (a
    module, a class instance, a dict, anything that isn't int/float/str/
    bool/list/None) — made tractable because Sense's own reference
    interpreter already runs inside the same Python process, not a
    separate one needing a real FFI. Member access and calls on a
    `PythonValue` delegate straight to the wrapped object.
    """

    value: Any
    label: str  # for type_of()/stringify()/error messages, e.g. "module 'math'"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<python {self.label}>"


def to_sense_value(value: Any) -> Any:
    """Converts a raw Python value coming back from a foreign call into a
    Sense value. Most Sense types already ARE these Python types (Int is
    `int`, String is `str`, ...), so real work only happens for values
    Sense has no native representation for."""
    if value is None or isinstance(value, (bool, int, float, str, PythonValue)):
        return value
    if isinstance(value, list):
        return [to_sense_value(v) for v in value]
    return PythonValue(value, label=type(value).__name__)


def from_sense_value(value: Any) -> Any:
    """The inverse, applied to arguments before a foreign call: unwrap a
    `PythonValue`. Every other Sense value already IS the Python value
    underneath, so there's nothing else to convert."""
    if isinstance(value, PythonValue):
        return value.value
    if isinstance(value, list):
        return [from_sense_value(v) for v in value]
    return value


def type_name(value: Any) -> str:
    if value is None:
        return "Nil"
    if isinstance(value, bool):
        return "Bool"
    if isinstance(value, int):
        return "Int"
    if isinstance(value, float):
        return "Float"
    if isinstance(value, str):
        return "String"
    if isinstance(value, list):
        return "Array"
    if isinstance(value, (SenseFunction, BuiltinFunction)):
        return "Function"
    if isinstance(value, SenseModule):
        return "Module"
    if isinstance(value, SenseModel):
        return "Model"
    if isinstance(value, Answer):
        return "Answer"
    if isinstance(value, SenseAgent):
        return "Agent"
    if isinstance(value, (SenseActionDef, SenseAction)):
        return "Action"
    if isinstance(value, SenseToolDef):
        return "Tool"
    if isinstance(value, (SenseMemory, SensePersistentMemory)):
        return "Memory"
    if isinstance(value, SenseSkill):
        return "Skill"
    if isinstance(value, McpTool):
        return "Tool"  # fills the same role a SenseToolDef does -- see McpTool's docstring
    if isinstance(value, SenseMcpServer):
        return "McpServer"
    if isinstance(value, SenseFuture):
        return "Future"
    if isinstance(value, PythonValue):
        return "Python"
    return type(value).__name__


def matches_type(value: Any, annotation: n.TypeAnnotation) -> bool:
    """Best-effort structural check of a runtime value against a declared type.

    This is intentionally shallow — Sense has no full static type inference
    yet, by design (a runtime check, not a type system in the PL-theory
    sense). It catches the common mistakes (wrong primitive kind, wrong
    array element type) without pretending to be a real type checker.
    """
    name = annotation.name
    if name == "Any":
        return True
    if name == "Int":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "Float":
        return isinstance(value, float) or (isinstance(value, int) and not isinstance(value, bool))
    if name == "String":
        return isinstance(value, str)
    if name == "Bool":
        return isinstance(value, bool)
    if name == "Nil":
        return value is None
    if name == "Array":
        if not isinstance(value, list):
            return False
        if annotation.params:
            elem_type = annotation.params[0]
            return all(matches_type(el, elem_type) for el in value)
        return True
    if name == "Function":
        return isinstance(value, (SenseFunction, BuiltinFunction))
    if name == "Model":
        return isinstance(value, SenseModel)
    if name == "Agent":
        return isinstance(value, SenseAgent)
    if name == "Action":
        return isinstance(value, (SenseActionDef, SenseAction))
    if name == "Tool":
        return isinstance(value, (SenseToolDef, McpTool))
    if name == "Memory":
        return isinstance(value, (SenseMemory, SensePersistentMemory))
    if name == "Skill":
        return isinstance(value, SenseSkill)
    if name == "McpServer":
        return isinstance(value, SenseMcpServer)
    if name == "Future":
        return isinstance(value, SenseFuture)
    if name == "Python":
        return isinstance(value, PythonValue)
    if name.lower() == "answer":
        if not isinstance(value, Answer):
            return False
        if annotation.params:
            return matches_type(value.value, annotation.params[0])
        return True
    # Unknown/custom type names are not enforced yet.
    return True


def is_truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return len(value) > 0
    if isinstance(value, list):
        return len(value) > 0
    return True


def stringify(value: Any) -> str:
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == int(value) and abs(value) < 1e15:
            return f"{value:.1f}"
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(stringify(v) for v in value) + "]"
    if isinstance(value, Answer):
        conf = "?" if value.confidence is None else f"{value.confidence:.2f}"
        return f"answer<{conf}>({stringify(value.value)})"
    if isinstance(value, SenseModel):
        return f"<model {value.name}>"
    if isinstance(value, SenseAgent):
        if value.status == "paused" and value.pause_reason:
            return f"<agent {value.name} status=paused ({value.pause_reason})>"
        return f"<agent {value.name} status={value.status}>"
    if isinstance(value, SenseActionDef):
        return f"<{value.kind} action {value.name}>"
    if isinstance(value, SenseAction):
        return f"<action {value.name} ({value.kind}) state={value.state}>"
    if isinstance(value, SenseToolDef):
        return f"<tool {value.name}>"
    if isinstance(value, SenseMemory):
        return f"<memory {value.name} ({len(value.store)} entries)>"
    if isinstance(value, SensePersistentMemory):
        return f"<memory {value.name} (persistent, {value.path})>"
    if isinstance(value, SenseSkill):
        return f"<skill {value.name} ({len(value.tools)} tools)>"
    if isinstance(value, McpTool):
        return f"<mcp tool {value.server.name}.{value.name}>"
    if isinstance(value, SenseMcpServer):
        return f"<mcp server {value.name} ({len(value.tools)} tools)>"
    if isinstance(value, SenseFuture):
        return f"<future {value.label} ({'ready' if value.done else 'pending'})>"
    if isinstance(value, PythonValue):
        try:
            return str(value.value)
        except Exception:  # noqa: BLE001 - a broken __str__ on the wrapped object shouldn't break print()
            return f"<python {value.label}>"
    return str(value)
