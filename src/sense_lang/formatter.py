"""Canonical pretty-printer: Sense source -> reformatted Sense source.

Parses to the AST and re-emits it with consistent indentation and spacing
— the AST already IS the exact grouping/precedence the parser produced, so
this is a minimal-parens expression printer plus a statement-by-statement
reconstruction of the surface syntax, not a transformation of meaning.
Canonicalization rules: always-multi-line blocks, `requires` ordering.
Not attempted: line-width wrapping, alignment.

Blank lines are *preserved*, not imposed: wherever the source had at least
one blank line immediately before a statement or comment, the output gets
exactly one (multiple consecutive blanks collapse to one; none stays
none). There's no rule forcing blank lines around functions/agents/etc. a
user didn't already separate — Sense's own minimum-lines goal argues
against a formatter that adds vertical space nobody asked for.

Comments are not part of the AST (the lexer discards them for the parser —
see lexer.py), so they're captured separately and re-attached here:
- a standalone comment (its own line in the source) is placed immediately
  before the next statement whose own line is >= the comment's line —
  a heuristic, not a guarantee of exact original position relative to
  blank lines around it.
- a trailing comment (followed real code on the same source line) is
  re-attached to that same statement's own primary output line — so
  `x = 5  # note` round-trips as itself, not hoisted above.
Either way, comment *text* is never silently dropped.
"""

from __future__ import annotations

from . import ast_nodes as n
from .lexer import Comment, tokenize_with_comments
from .parser import parse
from .values import stringify

INDENT = "    "

# Precedence levels, lowest to highest -- mirrors parser.py's grammar
# exactly (assignment -> or -> and -> equality -> comparison -> term ->
# factor -> unary -> call -> primary), used to print the minimal parens
# needed to preserve the AST's grouping.
_ASSIGN, _OR, _AND, _EQUALITY, _COMPARISON, _TERM, _FACTOR, _UNARY, _CALL, _PRIMARY = range(10)

_BINOP_PREC = {
    "==": _EQUALITY, "!=": _EQUALITY,
    "<": _COMPARISON, "<=": _COMPARISON, ">": _COMPARISON, ">=": _COMPARISON,
    "+": _TERM, "-": _TERM,
    "*": _FACTOR, "/": _FACTOR, "%": _FACTOR,
}

def format_source(source: str) -> str:
    tokens, comments = tokenize_with_comments(source)
    program = parse(tokens)
    blank_lines = {i + 1 for i, raw_line in enumerate(source.splitlines()) if raw_line.strip() == ""}
    return _Formatter(comments, blank_lines).format_program(program)


class _Formatter:
    def __init__(self, comments: list[Comment], blank_lines: set[int]):
        self._standalone = sorted((c for c in comments if not c.trailing), key=lambda c: c.line)
        self._standalone_idx = 0
        self._trailing: dict[int, str] = {c.line: c.text for c in comments if c.trailing}
        self._blank_lines = blank_lines
        self._out: list[str] = []

    def format_program(self, program: n.Program) -> str:
        self._format_statements(program.statements, 0)
        self._flush_standalone(float("inf"), 0)
        text = "\n".join(self._out)
        return text + "\n" if text else ""

    # -- comment placement ----------------------------------------------------

    def _flush_standalone(self, up_to_line: float, indent: int) -> None:
        while self._standalone_idx < len(self._standalone) and self._standalone[self._standalone_idx].line <= up_to_line:
            comment = self._standalone[self._standalone_idx]
            self._blank_before(comment.line)
            self._raw(indent, f"# {comment.text}" if comment.text else "#")
            self._standalone_idx += 1

    def _blank_before(self, line: int) -> None:
        """A blank line goes in the output iff the source had one
        immediately above this line — never imposed, never more than one."""
        if line > 1 and (line - 1) in self._blank_lines:
            self._ensure_blank()

    def _ensure_blank(self) -> None:
        if self._out and self._out[-1] != "":
            self._raw(0, "")

    def _trailing_suffix(self, line: int) -> str:
        text = self._trailing.pop(line, None)
        return f"  # {text}" if text else ""

    def _raw(self, indent: int, text: str) -> None:
        self._out.append(f"{INDENT * indent}{text}" if text else "")

    def _emit(self, indent: int, text: str, line: int) -> None:
        """Emit one statement's primary line, with its trailing comment
        (if any) reattached."""
        self._raw(indent, text + self._trailing_suffix(line))

    # -- statement lists -------------------------------------------------------

    def _format_statements(self, statements: list[n.Node], indent: int) -> None:
        for stmt in statements:
            self._flush_standalone(stmt.line, indent)
            self._blank_before(stmt.line)
            self._format_statement(stmt, indent)

    def _format_statement(self, stmt: n.Node, indent: int) -> None:
        method = getattr(self, f"_fmt_{type(stmt).__name__}", None)
        if method is None:
            raise NotImplementedError(f"formatter: no printer for statement {type(stmt).__name__}")
        method(stmt, indent)

    def _format_block(self, block: n.Block, indent: int) -> None:
        self._format_statements(block.statements, indent + 1)

    # -- statement printers -----------------------------------------------------

    def _params(self, params: list[n.Param]) -> str:
        return ", ".join(p.name + (f": {p.type_ann}" if p.type_ann else "") for p in params)

    def _fmt_TypedDecl(self, stmt: n.TypedDecl, indent: int) -> None:
        self._emit(indent, f"{stmt.name}: {stmt.type_ann} = {self._expr(stmt.value)}", stmt.line)

    def _fmt_LocalDecl(self, stmt: n.LocalDecl, indent: int) -> None:
        type_part = f": {stmt.type_ann}" if stmt.type_ann else ""
        self._emit(indent, f"local {stmt.name}{type_part} = {self._expr(stmt.value)}", stmt.line)

    def _fmt_FnDecl(self, stmt: n.FnDecl, indent: int) -> None:
        arrow = f" -> {stmt.return_type}" if stmt.return_type else ""
        self._emit(indent, f"def {stmt.name}({self._params(stmt.params)}){arrow}:", stmt.line)
        self._format_block(stmt.body, indent)

    def _fmt_TestStmt(self, stmt: n.TestStmt, indent: int) -> None:
        self._emit(indent, f'test "{_escape_string(stmt.description)}":', stmt.line)
        self._format_block(stmt.body, indent)

    def _fmt_SessionStmt(self, stmt: n.SessionStmt, indent: int) -> None:
        self._emit(indent, f"session {stmt.name}:", stmt.line)
        self._format_block(stmt.body, indent)

    def _fmt_AgentDecl(self, stmt: n.AgentDecl, indent: int) -> None:
        prefix = "async " if stmt.is_async else ""
        self._emit(indent, f"{prefix}agent {stmt.name}:", stmt.line)
        self._format_block(stmt.body, indent)

    def _fmt_ActionDecl(self, stmt: n.ActionDecl, indent: int) -> None:
        requirements = []
        if stmt.requires_capability is not None:
            requirements.append(stmt.requires_capability)
        if stmt.requires_approval:
            requirements.append("approval")
        requires = f" requires {', '.join(requirements)}" if requirements else ""
        self._emit(indent, f"{stmt.kind} action {stmt.name}({self._params(stmt.params)}){requires}:", stmt.line)
        self._format_block(stmt.body, indent)
        if stmt.rollback_body is not None:
            self._emit(indent, "rollback:", stmt.rollback_body.line)
            self._format_block(stmt.rollback_body, indent)

    def _fmt_ToolDecl(self, stmt: n.ToolDecl, indent: int) -> None:
        prefix = "async " if stmt.is_async else ""
        returns = f" returns {stmt.return_type}" if stmt.return_type else ""
        requires = f" requires {stmt.requires_capability}" if stmt.requires_capability else ""
        description = f' "{_escape_string(stmt.description)}"' if stmt.description is not None else ""
        self._emit(
            indent,
            f"{prefix}tool {stmt.name}({self._params(stmt.params)}){returns}{requires}{description}:",
            stmt.line,
        )
        self._format_block(stmt.body, indent)

    def _fmt_MemoryDecl(self, stmt: n.MemoryDecl, indent: int) -> None:
        prefix = "persistent " if stmt.is_persistent else ""
        kind = f': "{_escape_string(stmt.kind)}"' if stmt.kind is not None else ""
        self._emit(indent, f"{prefix}memory {stmt.name}{kind}", stmt.line)

    def _fmt_PolicyStmt(self, stmt: n.PolicyStmt, indent: int) -> None:
        if len(stmt.rules) == 1:
            rule = stmt.rules[0]
            self._emit(indent, f"policy: {rule.effect} {rule.capability}", rule.line)
            return
        self._emit(indent, "policy:", stmt.line)
        for rule in stmt.rules:
            self._emit(indent + 1, f"{rule.effect} {rule.capability}", rule.line)

    def _fmt_SetStmt(self, stmt: n.SetStmt, indent: int) -> None:
        self._emit(indent, f"set {stmt.name} = {self._expr(stmt.value)}", stmt.line)

    def _fmt_ModelDecl(self, stmt: n.ModelDecl, indent: int) -> None:
        self._emit(indent, f"model {stmt.name} = {self._expr(stmt.value)}", stmt.line)

    def _fmt_IfStmt(self, stmt: n.IfStmt, indent: int) -> None:
        self._emit(indent, f"if {self._expr(stmt.condition)}:", stmt.line)
        self._format_block(stmt.then_branch, indent)
        self._format_else_chain(stmt.else_branch, indent)

    def _format_else_chain(self, branch: "n.Block | n.IfStmt | None", indent: int) -> None:
        if branch is None:
            return
        # No _blank_before() here, deliberately: an `else`/`else if` stays
        # visually paired with its `if`, even if the source had a blank
        # line before it -- one of the rare cases where preservation would
        # hurt more than help.
        self._flush_standalone(branch.line, indent)
        if isinstance(branch, n.IfStmt):
            self._emit(indent, f"else if {self._expr(branch.condition)}:", branch.line)
            self._format_block(branch.then_branch, indent)
            self._format_else_chain(branch.else_branch, indent)
        else:
            self._emit(indent, "else:", branch.line)
            self._format_block(branch, indent)

    def _fmt_WhileStmt(self, stmt: n.WhileStmt, indent: int) -> None:
        self._emit(indent, f"while {self._expr(stmt.condition)}:", stmt.line)
        self._format_block(stmt.body, indent)

    def _fmt_ForStmt(self, stmt: n.ForStmt, indent: int) -> None:
        self._emit(indent, f"for {stmt.var_name} in {self._expr(stmt.iterable)}:", stmt.line)
        self._format_block(stmt.body, indent)

    def _fmt_ReturnStmt(self, stmt: n.ReturnStmt, indent: int) -> None:
        text = "return" if stmt.value is None else f"return {self._expr(stmt.value)}"
        self._emit(indent, text, stmt.line)

    def _fmt_BreakStmt(self, stmt: n.BreakStmt, indent: int) -> None:
        self._emit(indent, "break", stmt.line)

    def _fmt_ContinueStmt(self, stmt: n.ContinueStmt, indent: int) -> None:
        self._emit(indent, "continue", stmt.line)

    def _fmt_ImportStmt(self, stmt: n.ImportStmt, indent: int) -> None:
        alias = f" as {stmt.alias}" if stmt.alias else ""
        kind = "python " if stmt.is_python else ""
        self._emit(indent, f'import {kind}"{stmt.path}"{alias}', stmt.line)

    def _fmt_ExprStmt(self, stmt: n.ExprStmt, indent: int) -> None:
        self._emit(indent, self._expr(stmt.expr), stmt.line)

    # -- expressions: minimal-parens precedence printing -----------------------

    def _expr(self, expr: n.Node, min_prec: int = 0) -> str:
        text, prec = self._expr_and_prec(expr)
        return f"({text})" if prec < min_prec else text

    def _expr_and_prec(self, expr: n.Node) -> tuple[str, int]:
        method = getattr(self, f"_e_{type(expr).__name__}", None)
        if method is None:
            raise NotImplementedError(f"formatter: no printer for expression {type(expr).__name__}")
        return method(expr)

    def _e_IntLit(self, expr: n.IntLit):
        return stringify(expr.value), _PRIMARY

    def _e_FloatLit(self, expr: n.FloatLit):
        return stringify(expr.value), _PRIMARY

    def _e_StringLit(self, expr: n.StringLit):
        if "\n" in expr.value:
            # A single-quote pair already tolerates an embedded newline
            # (the idempotency guarantee doesn't require the source to
            # already use `\"\"\"`), but
            # _escape_string would flatten it into one line with literal
            # `\n` escapes -- semantically identical, but exactly wrong
            # for the case this exists for: a docstring. Printed back out
            # with `\"\"\"` and its actual line breaks intact instead.
            # Not reindented to match the block's current indent level --
            # a string's leading whitespace is part of its *value*, and
            # changing it would violate "formatting never changes program
            # behavior" for the sake of appearance.
            body = expr.value.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            return f'"""{body}"""', _PRIMARY
        return f'"{_escape_string(expr.value)}"', _PRIMARY

    def _e_BoolLit(self, expr: n.BoolLit):
        return stringify(expr.value), _PRIMARY

    def _e_NilLit(self, expr: n.NilLit):
        return "nil", _PRIMARY

    def _e_ArrayLit(self, expr: n.ArrayLit):
        return "[" + ", ".join(self._expr(el) for el in expr.elements) + "]", _PRIMARY

    def _e_Identifier(self, expr: n.Identifier):
        return expr.name, _PRIMARY

    def _e_Assign(self, expr: n.Assign):
        return f"{expr.name} = {self._expr(expr.value, _ASSIGN)}", _ASSIGN

    def _e_IndexAssign(self, expr: n.IndexAssign):
        target = self._expr(expr.target, _CALL)
        index = self._expr(expr.index)
        return f"{target}[{index}] = {self._expr(expr.value, _ASSIGN)}", _ASSIGN

    def _e_UnaryOp(self, expr: n.UnaryOp):
        operand, operand_prec = self._expr_and_prec(expr.operand)
        if operand_prec < _UNARY:
            operand = f"({operand})"
        if expr.op == "not":
            return f"not {operand}", _UNARY
        # avoid "--x" reading as a different operator
        sep = " " if operand.startswith("-") else ""
        return f"-{sep}{operand}", _UNARY

    def _e_LogicalOp(self, expr: n.LogicalOp):
        prec = _OR if expr.op == "or" else _AND
        left = self._expr(expr.left, prec)
        right = self._expr(expr.right, prec + 1)
        return f"{left} {expr.op} {right}", prec

    def _e_BinaryOp(self, expr: n.BinaryOp):
        prec = _BINOP_PREC[expr.op]
        left = self._expr(expr.left, prec)
        right = self._expr(expr.right, prec + 1)
        return f"{left} {expr.op} {right}", prec

    def _e_Call(self, expr: n.Call):
        callee = self._expr(expr.callee, _CALL)
        parts = [self._expr(a) for a in expr.args]
        parts += [f"{label}: {self._expr(v)}" for label, v in expr.kwargs]
        return f"{callee}({', '.join(parts)})", _CALL

    def _e_Index(self, expr: n.Index):
        target = self._expr(expr.target, _CALL)
        index = self._expr(expr.index)
        return f"{target}[{index}]", _CALL

    def _e_MemberAccess(self, expr: n.MemberAccess):
        target = self._expr(expr.target, _CALL)
        return f"{target}.{expr.name}", _CALL


def _escape_string(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
