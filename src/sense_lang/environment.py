"""Lexical scoping.

Also carries delegation (`model`), `policy`, and the enclosing `agent`, one
optional override slot per scope, resolved by walking the parent chain —
exactly the same mechanism as variable lookup. This replaced an earlier
design (a single `Interpreter`-global stack for delegation/policy) once
`agent` pause/resume needed it: a shared mutable stack breaks the moment
execution can hand control between two logically-independent contexts (a
paused agent and whatever resumed it), since "top of stack" no longer means
"the right scope." Attaching the state to `Environment` instead means each
context's data always lives with *its own* env chain, so nothing gets
confused about whose turn it is.
"""

from __future__ import annotations

from typing import Any

from .errors import SenseNameError


class Environment:
    __slots__ = ("vars", "parent", "model", "policy", "agent", "call_frame")

    def __init__(self, parent: "Environment | None" = None):
        self.vars: dict[str, Any] = {}
        self.parent = parent
        self.model: Any | None = None  # this scope's own `set delegation` override, if any
        self.policy: dict[str, bool] | None = None  # this scope's own `policy` rules, if any
        self.agent: Any | None = None  # set on an agent's own env at creation time
        self.call_frame: tuple[str, str] | None = None  # (kind, name) -- set on a tool/action call's own env

    def child(self) -> "Environment":
        return Environment(parent=self)

    def define(self, name: str, value: Any) -> None:
        self.vars[name] = value

    def get(self, name: str, line: int | None = None) -> Any:
        env: Environment | None = self
        while env is not None:
            if name in env.vars:
                return env.vars[name]
            env = env.parent
        raise SenseNameError(f"undefined variable '{name}'", line)

    def assign(self, name: str, value: Any, line: int | None = None) -> None:
        env: Environment | None = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        raise SenseNameError(f"cannot assign to undefined variable '{name}'", line)

    def define_or_assign(self, name: str, value: Any) -> None:
        """`name = value`: update the nearest existing binding in the scope
        chain, or create one in *this* (innermost) scope if none exists.

        This is Sense's one assignment rule. There is no separate
        declaration keyword and no Python-style implicit-local surprise:
        writing to a name that already exists anywhere in an enclosing
        scope updates it in place.
        """
        env: Environment | None = self
        while env is not None:
            if name in env.vars:
                env.vars[name] = value
                return
            env = env.parent
        self.vars[name] = value

    def has(self, name: str) -> bool:
        env: Environment | None = self
        while env is not None:
            if name in env.vars:
                return True
            env = env.parent
        return False

    # -- delegation / policy / agent context ---------------------------------

    def get_current_model(self) -> Any | None:
        env: Environment | None = self
        while env is not None:
            if env.model is not None:
                return env.model
            env = env.parent
        return None

    def set_current_model(self, model: Any) -> None:
        self.model = model

    def get_capability(self, capability: str) -> bool:
        """Default is allow -- a capability nobody's policy ever mentions
        is usable; `requires`/`policy` gate what's explicitly named, not
        everything by default."""
        env: Environment | None = self
        while env is not None:
            if env.policy is not None and capability in env.policy:
                return env.policy[capability]
            env = env.parent
        return True

    def set_policy_rule(self, capability: str, allowed: bool) -> None:
        if self.policy is None:
            self.policy = {}
        self.policy[capability] = allowed

    def get_current_agent(self) -> Any | None:
        env: Environment | None = self
        while env is not None:
            if env.agent is not None:
                return env.agent
            env = env.parent
        return None

    def get_lineage(self) -> list[str]:
        """Execution lineage for an audit event created at this scope:
        every enclosing agent and tool/action call boundary, outermost
        first -- e.g. `["agent:approver", "tool:search"]` for a tool
        called from inside that agent's body. Unlike `get_current_agent`
        (which stops at the *nearest* match), this walks the *whole*
        parent chain, since a call can nest inside several boundaries at
        once (an action's body calling a tool, itself inside an agent).

        Reuses the exact mechanism `.agent`/`.model`/`.policy` already
        use rather than a separate call-stack structure: each nested call
        already gets its own child `Environment` (`_call_tool`'s
        `call_env`, `_commit_action`'s `call_env`, ...), so marking one
        field on that child is enough -- and, like `.agent`, it can never
        get confused across two concurrent `async agent`/`async tool`
        executions, since each has its own independent env chain rather
        than sharing one mutable stack."""
        frames: list[str] = []
        env: Environment | None = self
        while env is not None:
            if env.agent is not None:
                frames.append(f"agent:{env.agent.name}")
            if env.call_frame is not None:
                kind, name = env.call_frame
                frames.append(f"{kind}:{name}")
            env = env.parent
        frames.reverse()
        return frames
        return None
