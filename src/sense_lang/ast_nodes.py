"""AST node definitions for Sense (deterministic core + probabilistic slice).

Node shapes are surface-syntax-independent from the grammar that produces
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class Node:
    line: int


# -- type annotations --------------------------------------------------------


@dataclass
class TypeAnnotation(Node):
    name: str
    params: list["TypeAnnotation"] = field(default_factory=list)
    line: int = 0

    def __str__(self) -> str:
        if self.params:
            inner = ", ".join(str(p) for p in self.params)
            return f"{self.name}<{inner}>"
        return self.name


# -- expressions --------------------------------------------------------------


@dataclass
class IntLit(Node):
    value: int
    line: int = 0


@dataclass
class FloatLit(Node):
    value: float
    line: int = 0


@dataclass
class StringLit(Node):
    value: str
    line: int = 0


@dataclass
class BoolLit(Node):
    value: bool
    line: int = 0


@dataclass
class NilLit(Node):
    line: int = 0


@dataclass
class ArrayLit(Node):
    elements: list[Node]
    line: int = 0


@dataclass
class Identifier(Node):
    name: str
    line: int = 0


@dataclass
class Assign(Node):
    name: str
    value: Node
    line: int = 0


@dataclass
class IndexAssign(Node):
    target: Node
    index: Node
    value: Node
    line: int = 0


@dataclass
class BinaryOp(Node):
    op: str
    left: Node
    right: Node
    line: int = 0


@dataclass
class LogicalOp(Node):
    op: str  # "and" or "or"
    left: Node
    right: Node
    line: int = 0


@dataclass
class UnaryOp(Node):
    op: str
    operand: Node
    line: int = 0


@dataclass
class Call(Node):
    callee: Node
    args: list[Node]
    # Trailing `label: expr` arguments (e.g. inference(..., temperature: 0.2))
    # -- deliberately narrower than general keyword arguments: only a
    # BuiltinFunction that opts in (accepts_kwargs=True) can receive these,
    # so this doesn't touch fn/tool/action param-binding at all. Always
    # empty for the overwhelming majority of calls.
    kwargs: list[tuple[str, Node]] = field(default_factory=list)
    line: int = 0


@dataclass
class Index(Node):
    target: Node
    index: Node
    line: int = 0


@dataclass
class MemberAccess(Node):
    target: Node
    name: str
    line: int = 0


# -- statements -----------------------------------------------------------------


@dataclass
class Param:
    name: str
    type_ann: TypeAnnotation | None = None


@dataclass
class TypedDecl(Node):
    """`name: Type = expr` — a declaration with an explicit type annotation.

    An untyped `name = expr` is *not* a separate AST node: it parses as a
    plain `ExprStmt` wrapping an `Assign`, and both forms resolve through
    the same "assign to the nearest existing binding, else create one in the
    current scope" rule at runtime.
    """

    name: str
    value: Node
    type_ann: TypeAnnotation
    line: int = 0


@dataclass
class LocalDecl(Node):
    """`local name [: Type] = expr` — always creates a *new* binding in the
    current scope, shadowing any same-named binding in an enclosing scope.

    This is the explicit escape hatch from the default `name = expr` /
    `TypedDecl` rule (nearest-existing-binding-or-create-here): use `local`
    when you deliberately want a fresh variable here, even if an outer one
    of the same name exists.
    """

    name: str
    value: Node
    type_ann: TypeAnnotation | None = None
    line: int = 0


@dataclass
class SetStmt(Node):
    name: str
    value: Node
    line: int = 0


@dataclass
class ModelDecl(Node):
    """`model name = expr` -- declares `name`, checked at binding time to
    actually evaluate to a `Model` (`SenseRuntimeError`/`SenseTypeError`
    otherwise). Not a new runtime concept -- same "declaring is just
    binding, with a check" shape `TypedDecl` already has for `x: Int =
    expr` -- but a real keyword, unlike `Skill`/plain `Model` values
    (`inference(...)`), which are deliberately NOT given dedicated
    declaration syntax elsewhere in Sense. `name` also
    becomes the model's own `.name` (overriding whatever the expression's
    own value carried), so `inference(...)` itself doesn't need a
    separate name argument -- the declaration already names it once.
    """

    name: str
    value: Node
    line: int = 0


@dataclass
class FnDecl(Node):
    """`def name(params) -> Type: ...` — a plain, deterministic function:
    no capability check, no audit entry, always runs immediately, never
    async. Deliberately matches Python's own function-declaration shape
    (`def`, `->`) rather than Sense's own house style everywhere else
    (`tool`/`action`/`agent`/... each keep their own leading keyword and
    `returns` for a return type) -- this is the one concept meant to be
    exactly what a Python programmer already knows, since a plain
    function has nothing to do with capabilities/safety/agency, the
    actual reasons the rest of the language reads differently from
    Python."""

    name: str
    params: list[Param]
    body: "Block"
    return_type: TypeAnnotation | None = None
    line: int = 0


@dataclass
class ActionDecl(Node):
    """`reversible action name(params) requires cap.path, approval: ...` —
    a world-changing operation, declared distinct from a plain `fn`.
    Calling it does NOT run the body: it produces a `SenseAction`
    ("prepared") — the full prepare -> verify -> commit law.

    `requires` accepts at most one capability path and/or the literal
    `approval`, in either order (`requires x.y, approval` or
    `requires approval, x.y`), closing the "declared approval marker" gap
    noted when `ask_human` shipped: an action can now *mandate* human
    approval before commit(), not just compose with ask_human() by hand.

    `rollback_body` is the compensation a reversible action must formally
    define -- declared with `rollback:` (originally
    `compensate:`, renamed to match the method that actually runs it:
    `rollback` is a plain IDENT here, not a keyword, matched by lexeme +
    a following ':' in `Parser._action_decl`. Reserving it as a real
    keyword no longer needs to preserve a bare `rollback(action)` call
    (that's gone -- `<action>.rollback()` is member access now), but a
    smaller ambiguity remains regardless: a bare top-level
    `rollback: Type = expr` typed declaration immediately after a
    reversible action's body would still be misread as that action's
    rollback block, since both start with IDENT ':'. Rare enough (using
    the name of a lifecycle method as a variable, at exactly that
    position) to keep accepting rather than reserve a word used in
    exactly one grammar position. Mandatory on
    `reversible` (the parser rejects a `reversible action` with no
    `rollback` block: a reversible action with no actual undo is a claim
    the declaration doesn't back up -- `irreversible` is the honest
    spelling for that), forbidden on `irreversible` -- so on this AST,
    `kind == "reversible"` and `rollback_body is not None` always agree;
    the type stays `Block | None` only because an `irreversible` action's
    is genuinely absent. `<action>.rollback()` runs this body after
    commit, in a scope nested inside the exact one `.commit()`'s body ran
    in -- so it sees the original args and any local variable that body
    computed.
    """

    name: str
    kind: str  # "reversible" | "irreversible"
    params: list[Param]
    body: "Block"
    requires_capability: str | None = None
    requires_approval: bool = False
    rollback_body: "Block | None" = None
    line: int = 0


@dataclass
class ToolDecl(Node):
    """`tool name(params) returns Type requires cap.path: ...` — a
    capability-checked function: external functionality with a typed
    interface and explicit permissions, distinct from `action`: calling
    it runs the body immediately, no prepare/verify/commit lifecycle. The
    capability (if any) is checked once at the call site against the
    caller's policy scope, before the body runs.

    `is_async`: `async tool ...` runs the body on its own thread and
    returns a `Future` immediately instead of running synchronously.

    `description`: an optional trailing string, e.g. `tool search(...)
    "search the web for a query":`. Backward compatible -- every tool
    keeps working undescribed exactly as before; it only becomes required
    at the point a tool is handed to `ask_with_tools(...)`, since LLM
    tool-calling lives or dies on a natural-language description of when
    to use it."""

    name: str
    params: list[Param]
    body: "Block"
    return_type: TypeAnnotation | None = None
    requires_capability: str | None = None
    is_async: bool = False
    description: str | None = None
    line: int = 0


@dataclass
class MemoryDecl(Node):
    """`memory name` / `memory name: "kind"` (session) or
    `persistent memory name` / `persistent memory name: "path"` (durable) —
    binds a memory store, distinct from an ordinary variable. The optional
    string means something different depending on `is_persistent`: for
    session memory it's a free-form, unvalidated hint
    (short-term/long-term/episodic/semantic/working are *potential*
    categories, not a closed set) kept only for introspection; for
    persistent memory it's the SQLite file path to store it in (see
    `Interpreter._default_memory_path` for where it lands if omitted --
    not simply the current working directory, since that resolved
    differently depending on which directory `sense run` happened to be
    invoked from)."""

    name: str
    kind: str | None = None
    is_persistent: bool = False
    line: int = 0


@dataclass
class PolicyRule:
    effect: str  # "allow" | "deny"
    capability: str
    line: int = 0


@dataclass
class PolicyStmt(Node):
    rules: list[PolicyRule]
    line: int = 0


@dataclass
class TestStmt(Node):
    """`test "description": ...` — an isolated, individually-reported
    assertion block (Phase 5: Developer Experience). Runs in its own child
    scope; a failure inside is caught and recorded rather than crashing the
    rest of the file, so multiple `test` blocks in one file each get their
    own pass/fail result. See `sense test` in the CLI reference."""

    description: str
    body: "Block"
    line: int = 0


@dataclass
class SessionStmt(Node):
    """`session name: ...` — an execution/context boundary, not an
    autonomous actor. Runs its body immediately in a fresh child scope
    with its own delegation frame (`set delegation = ...` inside does not
    leak to the surrounding scope)."""

    name: str
    body: "Block"
    line: int = 0


@dataclass
class AgentDecl(Node):
    """`agent name: ...` — declares (but does not run) an autonomous
    process: identity + a persistent scope + a lifecycle. Declaring one only
    reaches "created"; the builtin `start(agent)` runs its body once,
    advancing it to "running" then "completed"/"failed".

    `is_async`: `async agent ...` makes `start()`/`resume()` return
    immediately instead of blocking until the body settles — the caller and
    the agent's body can then genuinely be executing at the same time."""

    name: str
    body: "Block"
    is_async: bool = False
    line: int = 0


@dataclass
class IfStmt(Node):
    condition: Node
    then_branch: "Block"
    else_branch: "Block | IfStmt | None"
    line: int = 0


@dataclass
class WhileStmt(Node):
    condition: Node
    body: "Block"
    line: int = 0


@dataclass
class ForStmt(Node):
    var_name: str
    iterable: Node
    body: "Block"
    line: int = 0


@dataclass
class ReturnStmt(Node):
    value: Node | None
    line: int = 0


@dataclass
class BreakStmt(Node):
    line: int = 0


@dataclass
class ContinueStmt(Node):
    line: int = 0


@dataclass
class ImportStmt(Node):
    """`import "./file.sns" as name` (a Sense module) or
    `import python "module.name" as name` (Python interop — calls
    straight into an installed Python module). `is_python` distinguishes
    them; `path` is a relative file path for the former, a dotted Python
    module name for the latter.

    The quoted-string form is always accepted; a bare dotted identifier
    (`import pkg.mod as m`) is sugar for it, resolved to the same `path`
    string at parse time (`Parser._import_path`) -- `pkg.mod` becomes
    `"pkg/mod.sns"` for a Sense module, `"pkg.mod"` (unchanged) for
    `import python`. No separate AST shape or interpreter handling needed
    for the bare form; by the time this node exists, both spellings have
    already collapsed to the same string."""

    path: str
    alias: str | None
    is_python: bool = False
    line: int = 0


@dataclass
class FromImportStmt(Node):
    """`from "./file.sns" import a, b as c` / `from pkg.mod import a` (a
    Sense module) or `from python "module.name" import a, b` (Python
    interop) -- pulls specific names directly into the current scope,
    instead of binding the whole module under one alias the way `import`
    does. `path`/`is_python` mean exactly what they mean on `ImportStmt`
    (same `Parser._import_path` resolution, quoted or bare). `names` is a
    list of `(original_name, alias_or_None)` pairs, evaluated by loading
    the module once (same cache `import` uses -- see
    `Interpreter._load_module`/`_load_python_module`) and then binding
    each requested name directly, checked against the module's own
    members (`SenseModule.env`/`getattr` on the wrapped Python object) --
    a name that isn't actually there raises immediately, the same
    "fail at the import, not at first use" guarantee `import` already
    gives for a missing file."""

    path: str
    names: list[tuple[str, str | None]]
    is_python: bool = False
    line: int = 0


@dataclass
class ExprStmt(Node):
    expr: Node
    line: int = 0


@dataclass
class Block(Node):
    statements: list[Node]
    line: int = 0


@dataclass
class Program(Node):
    statements: list[Node]
    line: int = 0
