"""Recursive-descent parser: list[Token] -> Program AST.

Grammar (informal EBNF) below. Blocks are indentation, not braces:
NEWLINE/INDENT/DEDENT tokens come from the lexer, so this grammar looks a
lot like Python's statement grammar even though the keywords differ.

    program     := statement* EOF
    statement   := typed_decl | local_decl | fn_decl | test_stmt
                 | session_stmt | agent_decl | action_decl | tool_decl
                 | memory_decl | model_decl | policy_stmt
                 | set_stmt | if_stmt | while_stmt | for_stmt | return_stmt
                 | break_stmt | continue_stmt | import_stmt
                 | from_import_stmt | expr_stmt

    typed_decl  := IDENT ":" type "=" expr NEWLINE
    local_decl  := "local" IDENT (":" type)? "=" expr NEWLINE
    fn_decl     := "def" IDENT "(" params? ")" ("->" type)? block
                   # deliberately Python's own shape ("def"/"->"), unlike
                   # every other declaration below -- see _fn_decl
    model_decl  := "model" IDENT "=" expr NEWLINE
    test_stmt   := "test" STRING block
    session_stmt := "session" IDENT block
    agent_decl  := "async"? "agent" IDENT block
    action_decl := ("reversible" | "irreversible") "action" IDENT
                   "(" params? ")" ("requires" requirement ("," requirement)*)? block
                   ("rollback" block)?              # required on "reversible",
                                                     # forbidden on "irreversible"
                                                     # -- "rollback" here is the
                                                     # plain IDENT `rollback`
                                                     # (not a keyword), matched
                                                     # by lexeme + a following
                                                     # ':' -- see _action_decl
                                                     # for the ambiguity this
                                                     # still leaves
    tool_decl   := "async"? "tool" IDENT "(" params? ")" ("returns" type)?
                   ("requires" capability_path)? STRING? block
    memory_decl := "persistent"? "memory" IDENT (":" STRING)? NEWLINE
    requirement := "approval" | capability_path
    capability_path := IDENT ("." IDENT)*
    policy_stmt := "policy" ":" (NEWLINE INDENT policy_rule+ DEDENT | policy_rule)
    policy_rule := ("allow" | "deny") capability_path NEWLINE
    params      := param ("," param)*
    param       := IDENT (":" type)?
    type        := IDENT ("<" type ("," type)* ">")?

    set_stmt    := "set" IDENT "=" expr NEWLINE
    if_stmt     := "if" expr block ("else" (if_stmt | block))?
    while_stmt  := "while" expr block
    for_stmt    := "for" IDENT "in" expr block
    return_stmt := "return" expr? NEWLINE
    break_stmt  := "break" NEWLINE
    continue_stmt := "continue" NEWLINE
    import_stmt := "import" "python"? module_ref ("as" IDENT)? NEWLINE
    from_import_stmt := "from" "python"? module_ref "import"
                         import_name ("," import_name)* NEWLINE
    import_name := IDENT ("as" IDENT)?
    module_ref  := STRING | IDENT ("." IDENT)*
                 # bare form is sugar, resolved to the same string a
                 # quoted path would be, at parse time (_import_path):
                 # dots become "/" + ".sns" appended for a Sense module
                 # ("import pkg.mod" -> "pkg/mod.sns"), left as a dotted
                 # name unchanged for "import python" ("import os.path"
                 # -> "os.path"). The quoted form still works everywhere
                 # -- needed for a path/name a bare identifier chain can't
                 # spell (e.g. "../shared/utils.sns", or a Python package
                 # whose name isn't a valid identifier).
    expr_stmt   := expr NEWLINE

    block       := ":" NEWLINE INDENT statement+ DEDENT   # multi-line
                 | ":" statement                          # inline, e.g. `if x: return y`

    expr        := assignment
    assignment  := (IDENT | index_expr) "=" assignment | logic_or
    logic_or    := logic_and ("or" logic_and)*
    logic_and   := equality ("and" equality)*
    equality    := comparison (("==" | "!=") comparison)*
    comparison  := term (("<" | "<=" | ">" | ">=") term)*
    term        := factor (("+" | "-") factor)*
    factor      := unary (("*" | "/" | "%") unary)*
    unary       := ("not" | "-") unary | call
    call        := primary ( "(" args ")" | "." IDENT | "[" expr "]" )*
    args        := (arg ("," arg)*)?
    arg         := (IDENT ":" expr) | expr   # labeled must follow all positional;
                                              # only opted-in builtins accept any
    primary     := INT | FLOAT | STRING | "true" | "false" | "nil"
                 | IDENT | "(" expr ")" | "[" (expr ("," expr)*)? "]"
"""

from __future__ import annotations

from . import ast_nodes as n
from .errors import SenseSyntaxError
from .tokens import KEYWORDS, Token, TokenType as T

_KEYWORD_TOKEN_TYPES = frozenset(KEYWORDS.values())


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- token helpers -----------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[idx]

    def _previous(self) -> Token:
        return self.tokens[self.pos - 1]

    def _at_end(self) -> bool:
        return self._peek().type == T.EOF

    def _advance(self) -> Token:
        if not self._at_end():
            self.pos += 1
        return self._previous()

    def _check(self, type_: T) -> bool:
        return not self._at_end() and self._peek().type == type_

    def _match(self, *types: T) -> bool:
        if self._peek().type in types:
            self._advance()
            return True
        return False

    def _expect(self, type_: T, message: str) -> Token:
        if self._check(type_):
            return self._advance()
        raise SenseSyntaxError(
            f"{message} (got {self._peek().type.name} {self._peek().lexeme!r})",
            self._peek().line,
        )

    def _end_of_statement(self) -> None:
        self._expect(T.NEWLINE, "expected a new line after this statement")

    # -- entry point ---------------------------------------------------------

    def parse_program(self) -> n.Program:
        statements = []
        while not self._at_end():
            statements.append(self._statement())
        return n.Program(statements=statements, line=1)

    # -- statements ------------------------------------------------------------

    def _looks_like_fn_decl_missing_def(self) -> bool:
        """IDENT '(' ... matching ')' then (':' | 'returns') with no leading
        'def' -- a function declaration written the old way (an earlier
        version of Sense had no 'def' at all), not a call expression like
        `foo(1, 2)`: a bare call
        as a statement is never followed by ':'/'returns' in valid Sense,
        so this is unambiguous. Used only to raise a clear, specific error
        pointing at 'def' rather than a generic parse failure -- the same
        courtesy the lexer's own -> and ! guards already give."""
        if self._peek().type != T.IDENT or self._peek(1).type != T.LPAREN:
            return False
        depth = 0
        j = self.pos + 1
        while j < len(self.tokens):
            t = self.tokens[j].type
            if t == T.LPAREN:
                depth += 1
            elif t == T.RPAREN:
                depth -= 1
                if depth == 0:
                    break
            elif t == T.EOF:
                return False
            j += 1
        else:
            return False
        nxt = self.tokens[j + 1].type if j + 1 < len(self.tokens) else T.EOF
        return nxt in (T.COLON, T.RETURNS)

    def _statement(self) -> n.Node:
        if self._match(T.DEF):
            return self._fn_decl()
        if self._check(T.IDENT) and self._looks_like_fn_decl_missing_def():
            raise SenseSyntaxError(
                "function declarations need 'def' -- e.g. 'def "
                f"{self._peek().lexeme}(...) -> Type:'",
                self._peek().line,
            )
        if self._check(T.IDENT) and self._peek(1).type == T.COLON:
            return self._typed_decl()
        if self._match(T.LOCAL):
            return self._local_decl()
        if self._match(T.TEST):
            return self._test_stmt()
        if self._match(T.SESSION):
            return self._session_stmt()
        if self._match(T.AGENT):
            return self._agent_decl()
        if self._match(T.REVERSIBLE):
            return self._action_decl("reversible")
        if self._match(T.IRREVERSIBLE):
            return self._action_decl("irreversible")
        if self._match(T.TOOL):
            return self._tool_decl()
        if self._match(T.ASYNC):
            if self._match(T.AGENT):
                return self._agent_decl(is_async=True)
            if self._match(T.TOOL):
                return self._tool_decl(is_async=True)
            raise SenseSyntaxError("expected 'agent' or 'tool' after 'async'", self._peek().line)
        if self._match(T.MEMORY):
            return self._memory_decl()
        if self._match(T.PERSISTENT):
            self._expect(T.MEMORY, "expected 'memory' after 'persistent'")
            return self._memory_decl(is_persistent=True)
        if self._match(T.POLICY):
            return self._policy_stmt()
        if self._match(T.SET):
            return self._set_stmt()
        if self._match(T.MODEL):
            return self._model_decl()
        if self._match(T.IF):
            return self._if_stmt()
        if self._match(T.WHILE):
            return self._while_stmt()
        if self._match(T.FOR):
            return self._for_stmt()
        if self._match(T.RETURN):
            return self._return_stmt()
        if self._match(T.BREAK):
            line = self._previous().line
            self._end_of_statement()
            return n.BreakStmt(line=line)
        if self._match(T.CONTINUE):
            line = self._previous().line
            self._end_of_statement()
            return n.ContinueStmt(line=line)
        if self._match(T.IMPORT):
            return self._import_stmt()
        if self._match(T.FROM):
            return self._from_import_stmt()
        return self._expr_stmt()

    def _typed_decl(self) -> n.TypedDecl:
        name_tok = self._expect(T.IDENT, "expected a name")
        self._expect(T.COLON, "expected ':' for a type annotation")
        type_ann = self._type_annotation()
        self._expect(T.EQ, "expected '=' after the type annotation")
        value = self._expr()
        self._end_of_statement()
        return n.TypedDecl(name=name_tok.lexeme, value=value, type_ann=type_ann, line=name_tok.line)

    def _local_decl(self) -> n.LocalDecl:
        line = self._previous().line
        name_tok = self._expect(T.IDENT, "expected a name after 'local'")
        type_ann = None
        if self._match(T.COLON):
            type_ann = self._type_annotation()
        self._expect(T.EQ, "expected '=' in a local declaration")
        value = self._expr()
        self._end_of_statement()
        return n.LocalDecl(name=name_tok.lexeme, value=value, type_ann=type_ann, line=line)

    def _test_stmt(self) -> n.TestStmt:
        line = self._previous().line
        desc_tok = self._expect(T.STRING, "expected a description string after 'test'")
        body = self._block()
        return n.TestStmt(description=desc_tok.literal, body=body, line=line)

    def _session_stmt(self) -> n.SessionStmt:
        line = self._previous().line
        name = self._expect(T.IDENT, "expected a session name after 'session'").lexeme
        body = self._block()
        return n.SessionStmt(name=name, body=body, line=line)

    def _agent_decl(self, is_async: bool = False) -> n.AgentDecl:
        line = self._previous().line
        name = self._expect(T.IDENT, "expected an agent name after 'agent'").lexeme
        body = self._block()
        return n.AgentDecl(name=name, body=body, is_async=is_async, line=line)

    def _action_decl(self, kind: str) -> n.ActionDecl:
        self._expect(T.ACTION, f"expected 'action' after '{kind}'")
        name_tok = self._expect(T.IDENT, "expected an action name")
        self._expect(T.LPAREN, "expected '(' after the action name")
        params: list[n.Param] = []
        if not self._check(T.RPAREN):
            while True:
                pname = self._expect(T.IDENT, "expected a parameter name").lexeme
                ptype = None
                if self._match(T.COLON):
                    ptype = self._type_annotation()
                params.append(n.Param(name=pname, type_ann=ptype))
                if not self._match(T.COMMA):
                    break
        self._expect(T.RPAREN, "expected ')' after the parameters")
        requires_capability = None
        requires_approval = False
        if self._match(T.REQUIRES):
            while True:
                if self._match(T.APPROVAL):
                    if requires_approval:
                        raise SenseSyntaxError("'approval' listed twice in 'requires'", self._previous().line)
                    requires_approval = True
                elif self._check(T.IDENT) or self._peek().type in _KEYWORD_TOKEN_TYPES:
                    if requires_capability is not None:
                        raise SenseSyntaxError("an action can only 'require' one capability", self._peek().line)
                    requires_capability = self._capability_path()
                else:
                    raise SenseSyntaxError(
                        "expected a capability name or 'approval' after 'requires'", self._peek().line
                    )
                if not self._match(T.COMMA):
                    break
        body = self._block()
        rollback_body = None
        # 'rollback' here is a plain IDENT, not a keyword -- recognized
        # only by lexeme + a following ':' (never '('), the same
        # minimal-lookahead disambiguation `typed_decl` already uses for
        # IDENT ':'. The body is run via `<action>.rollback()` (member
        # access, not a bare call), so there's no builtin-call syntax left
        # to preserve by staying a plain IDENT -- but a smaller ambiguity
        # remains regardless: a bare top-level `rollback: Type = expr`
        # typed declaration right after a reversible action's body would
        # still be misread as this block. Rare enough to accept.
        if self._check(T.IDENT) and self._peek().lexeme == "rollback" and self._peek(1).type == T.COLON:
            if kind != "reversible":
                raise SenseSyntaxError(
                    "only a 'reversible action' may declare 'rollback' — "
                    "an irreversible action cannot be reliably undone",
                    self._peek().line,
                )
            self._advance()
            rollback_body = self._block()
        elif self._check(T.IDENT) and self._peek().lexeme == "compensate" and self._peek(1).type == T.COLON:
            # Old name for this block, kept detectable on sight for a
            # specific, helpful error rather than a generic parse failure
            # -- same courtesy `def` gets over the pre-`def` fn syntax.
            raise SenseSyntaxError(
                "'compensate' is now spelled 'rollback' -- rename this block's "
                "leading word to 'rollback:'",
                self._peek().line,
            )
        if kind == "reversible" and rollback_body is None:
            # Mandatory, not just permitted: a `reversible action` with no
            # actual undo is a claim ("this can be reliably undone") the
            # declaration itself doesn't back up -- `irreversible` is the
            # honest spelling for "no rollback," so this is caught here
            # rather than left to be a silent, easy-to-forget gap that only
            # surfaces later as a confusing "no 'rollback' block" runtime
            # error the first time someone actually calls .rollback().
            raise SenseSyntaxError(
                f"reversible action '{name_tok.lexeme}' must declare a 'rollback' block -- "
                "use 'irreversible action' instead if this genuinely can't be undone",
                self._peek().line,
            )
        return n.ActionDecl(
            name=name_tok.lexeme,
            kind=kind,
            params=params,
            body=body,
            requires_capability=requires_capability,
            requires_approval=requires_approval,
            rollback_body=rollback_body,
            line=name_tok.line,
        )

    def _capability_segment(self, message: str) -> str:
        """A capability path (`model.anthropic`, `payment.execute`, ...)
        is a free-form dotted namespace label, not a Sense identifier --
        it names an external resource/permission, never binds a variable
        -- so reserving a word as a language keyword (`model`, `action`,
        `tool`, `set`, ...) must not retroactively make it illegal to use
        in one. Accepts a plain IDENT or any keyword token's own lexeme
        (e.g. the `MODEL` token from the word "model") interchangeably."""
        if self._check(T.IDENT) or self._peek().type in _KEYWORD_TOKEN_TYPES:
            return self._advance().lexeme
        raise SenseSyntaxError(f"{message} (got {self._peek().type.name} {self._peek().lexeme!r})", self._peek().line)

    def _capability_path(self) -> str:
        parts = [self._capability_segment("expected a capability name")]
        while self._match(T.DOT):
            parts.append(self._capability_segment("expected a name after '.'"))
        return ".".join(parts)

    def _tool_decl(self, is_async: bool = False) -> n.ToolDecl:
        name_tok = self._expect(T.IDENT, "expected a tool name")
        self._expect(T.LPAREN, "expected '(' after the tool name")
        params: list[n.Param] = []
        if not self._check(T.RPAREN):
            while True:
                pname = self._expect(T.IDENT, "expected a parameter name").lexeme
                ptype = None
                if self._match(T.COLON):
                    ptype = self._type_annotation()
                params.append(n.Param(name=pname, type_ann=ptype))
                if not self._match(T.COMMA):
                    break
        self._expect(T.RPAREN, "expected ')' after the parameters")
        return_type = None
        if self._match(T.RETURNS):
            return_type = self._type_annotation()
        requires_capability = None
        if self._match(T.REQUIRES):
            requires_capability = self._capability_path()
        description = None
        if self._check(T.STRING):
            description = self._advance().literal
        body = self._block()
        return n.ToolDecl(
            name=name_tok.lexeme,
            params=params,
            body=body,
            return_type=return_type,
            requires_capability=requires_capability,
            is_async=is_async,
            description=description,
            line=name_tok.line,
        )

    def _memory_decl(self, is_persistent: bool = False) -> n.MemoryDecl:
        line = self._previous().line
        name_tok = self._expect(T.IDENT, "expected a memory name after 'memory'")
        kind = None
        if self._match(T.COLON):
            kind_tok = self._expect(T.STRING, "expected a string after ':' in a memory declaration")
            kind = kind_tok.literal
        self._end_of_statement()
        return n.MemoryDecl(name=name_tok.lexeme, kind=kind, is_persistent=is_persistent, line=line)

    def _policy_stmt(self) -> n.PolicyStmt:
        line = self._previous().line
        self._expect(T.COLON, "expected ':' after 'policy'")
        rules: list[n.PolicyRule] = []
        if self._match(T.NEWLINE):
            self._expect(T.INDENT, "expected an indented block of policy rules")
            while not self._check(T.DEDENT) and not self._at_end():
                rules.append(self._policy_rule())
            self._expect(T.DEDENT, "expected the policy block to end")
        else:
            rules.append(self._policy_rule())
        return n.PolicyStmt(rules=rules, line=line)

    def _policy_rule(self) -> n.PolicyRule:
        if self._match(T.ALLOW):
            effect = "allow"
        elif self._match(T.DENY):
            effect = "deny"
        else:
            raise SenseSyntaxError("expected 'allow' or 'deny' in a policy rule", self._peek().line)
        line = self._previous().line
        capability = self._capability_path()
        self._end_of_statement()
        return n.PolicyRule(effect=effect, capability=capability, line=line)

    def _fn_decl(self) -> n.FnDecl:
        """Called with 'def' already consumed by `_statement()`. Deliberately
        matches Python's own `def name(params) -> Type:` shape -- `def` as
        the leading keyword, `->` for the return type -- unlike every other
        declaration in Sense (`tool`/`action`/`agent`/...), which keep their
        own distinct keyword-first shape and `returns` for a return type.
        This one concept is meant to be exactly what a Python programmer
        already knows, not a Sense-flavored variant of it."""
        name_tok = self._expect(T.IDENT, "expected a function name after 'def'")
        self._expect(T.LPAREN, "expected '(' after the function name")
        params: list[n.Param] = []
        if not self._check(T.RPAREN):
            while True:
                pname = self._expect(T.IDENT, "expected a parameter name").lexeme
                ptype = None
                if self._match(T.COLON):
                    ptype = self._type_annotation()
                params.append(n.Param(name=pname, type_ann=ptype))
                if not self._match(T.COMMA):
                    break
        self._expect(T.RPAREN, "expected ')' after the parameters")
        return_type = None
        if self._match(T.ARROW):
            return_type = self._type_annotation()
        body = self._block()
        return n.FnDecl(name=name_tok.lexeme, params=params, body=body, return_type=return_type, line=name_tok.line)

    def _type_annotation(self) -> n.TypeAnnotation:
        tok = self._expect(T.IDENT, "expected a type name")
        params: list[n.TypeAnnotation] = []
        if self._match(T.LT):
            params.append(self._type_annotation())
            while self._match(T.COMMA):
                params.append(self._type_annotation())
            self._expect(T.GT, "expected '>' to close the generic type")
        return n.TypeAnnotation(name=tok.lexeme, params=params, line=tok.line)

    def _set_stmt(self) -> n.SetStmt:
        line = self._previous().line
        name = self._expect(T.IDENT, "expected a configuration name after 'set'").lexeme
        self._expect(T.EQ, "expected '=' in a set statement")
        value = self._expr()
        self._end_of_statement()
        return n.SetStmt(name=name, value=value, line=line)

    def _model_decl(self) -> n.ModelDecl:
        line = self._previous().line
        name = self._expect(T.IDENT, "expected a name after 'model'").lexeme
        self._expect(T.EQ, "expected '=' in a model declaration")
        value = self._expr()
        self._end_of_statement()
        return n.ModelDecl(name=name, value=value, line=line)

    def _if_stmt(self) -> n.IfStmt:
        line = self._previous().line
        condition = self._expr()
        then_branch = self._block()
        else_branch = None
        if self._match(T.ELSE):
            if self._match(T.IF):
                else_branch = self._if_stmt()
            else:
                else_branch = self._block()
        return n.IfStmt(condition=condition, then_branch=then_branch, else_branch=else_branch, line=line)

    def _while_stmt(self) -> n.WhileStmt:
        line = self._previous().line
        condition = self._expr()
        body = self._block()
        return n.WhileStmt(condition=condition, body=body, line=line)

    def _for_stmt(self) -> n.ForStmt:
        line = self._previous().line
        var_name = self._expect(T.IDENT, "expected a loop variable name after 'for'").lexeme
        self._expect(T.IN, "expected 'in' after the loop variable")
        iterable = self._expr()
        body = self._block()
        return n.ForStmt(var_name=var_name, iterable=iterable, body=body, line=line)

    def _return_stmt(self) -> n.ReturnStmt:
        line = self._previous().line
        value = None
        if not self._check(T.NEWLINE):
            value = self._expr()
        self._end_of_statement()
        return n.ReturnStmt(value=value, line=line)

    def _import_stmt(self) -> n.ImportStmt:
        line = self._previous().line
        is_python = self._match(T.PYTHON)
        path = self._import_path(is_python)
        alias = None
        if self._match(T.AS):
            alias = self._expect(T.IDENT, "expected an alias name after 'as'").lexeme
        self._end_of_statement()
        return n.ImportStmt(path=path, alias=alias, is_python=is_python, line=line)

    def _from_import_stmt(self) -> n.FromImportStmt:
        line = self._previous().line
        is_python = self._match(T.PYTHON)
        path = self._import_path(is_python)
        self._expect(T.IMPORT, "expected 'import' after the module in a 'from' statement")
        names: list[tuple[str, str | None]] = []
        while True:
            name = self._expect(T.IDENT, "expected a name to import").lexeme
            alias = None
            if self._match(T.AS):
                alias = self._expect(T.IDENT, "expected an alias name after 'as'").lexeme
            names.append((name, alias))
            if not self._match(T.COMMA):
                break
        self._end_of_statement()
        return n.FromImportStmt(path=path, names=names, is_python=is_python, line=line)

    def _import_path(self, is_python: bool) -> str:
        """`module_ref := STRING | IDENT ("." IDENT)*` -- both spellings
        collapse to the same string here, so nothing downstream (AST,
        interpreter) needs to know which one was written. A quoted string
        is used exactly as given. A bare dotted identifier chain is sugar:
        for `import python`, the dots already are the Python module
        separator, so it's joined back unchanged ("os.path" whether typed
        bare or quoted); for a Sense module, the dots become path
        separators and ".sns" is appended ("pkg.mod" -> "pkg/mod.sns"),
        matching how a Sense module is actually laid out on disk."""
        if self._check(T.STRING):
            return self._advance().literal
        first = self._expect(
            T.IDENT,
            "expected a Python module name (a string, or a bare dotted name like os.path) after 'import python'"
            if is_python
            else "expected a module path (a string, or a bare dotted name like pkg.mod) after 'import'",
        ).lexeme
        parts = [first]
        while self._check(T.DOT) and self._peek(1).type == T.IDENT:
            self._advance()  # consume '.'
            parts.append(self._advance().lexeme)
        if is_python:
            return ".".join(parts)
        return "/".join(parts) + ".sns"

    def _expr_stmt(self) -> n.ExprStmt:
        line = self._peek().line
        expr = self._expr()
        self._end_of_statement()
        return n.ExprStmt(expr=expr, line=line)

    def _block(self) -> n.Block:
        line = self._expect(T.COLON, "expected ':' to start a block").line
        if self._match(T.NEWLINE):
            self._expect(T.INDENT, "expected an indented block")
            statements = []
            while not self._check(T.DEDENT) and not self._at_end():
                statements.append(self._statement())
            self._expect(T.DEDENT, "expected the indented block to end")
            return n.Block(statements=statements, line=line)
        # inline form: `if x: return y`
        return n.Block(statements=[self._statement()], line=line)

    # -- expressions -----------------------------------------------------------

    def _expr(self) -> n.Node:
        return self._assignment()

    def _assignment(self) -> n.Node:
        expr = self._logic_or()
        if self._match(T.EQ):
            line = self._previous().line
            value = self._assignment()
            if isinstance(expr, n.Identifier):
                # `model NAME = inference(...)` is the required spelling for
                # naming a Model -- a plain `NAME = inference(...)` (no
                # keyword) would silently produce the exact same runtime
                # value with no name of its own, so it's caught here at
                # parse time rather than left as a quietly-worse-but-legal
                # alternative. Only the direct-call shape is checked (not
                # e.g. `x = some_var_holding_a_model`) -- this is about
                # *minting* a model from inference(...), not every place a
                # Model value can flow through a plain variable.
                if (
                    isinstance(value, n.Call)
                    and isinstance(value.callee, n.Identifier)
                    and value.callee.name == "inference"
                ):
                    raise SenseSyntaxError(
                        f"'{expr.name} = inference(...)' needs the 'model' keyword to name it -- "
                        f"use 'model {expr.name} = inference(...)' instead",
                        line,
                    )
                return n.Assign(name=expr.name, value=value, line=line)
            if isinstance(expr, n.Index):
                return n.IndexAssign(target=expr.target, index=expr.index, value=value, line=line)
            raise SenseSyntaxError("invalid assignment target", line)
        return expr

    def _logic_or(self) -> n.Node:
        expr = self._logic_and()
        while self._match(T.OR):
            line = self._previous().line
            right = self._logic_and()
            expr = n.LogicalOp(op="or", left=expr, right=right, line=line)
        return expr

    def _logic_and(self) -> n.Node:
        expr = self._equality()
        while self._match(T.AND):
            line = self._previous().line
            right = self._equality()
            expr = n.LogicalOp(op="and", left=expr, right=right, line=line)
        return expr

    def _equality(self) -> n.Node:
        expr = self._comparison()
        while self._match(T.EQEQ, T.BANGEQ):
            op_tok = self._previous()
            right = self._comparison()
            expr = n.BinaryOp(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _comparison(self) -> n.Node:
        expr = self._term()
        while self._match(T.LT, T.LTEQ, T.GT, T.GTEQ):
            op_tok = self._previous()
            right = self._term()
            expr = n.BinaryOp(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _term(self) -> n.Node:
        expr = self._factor()
        while self._match(T.PLUS, T.MINUS):
            op_tok = self._previous()
            right = self._factor()
            expr = n.BinaryOp(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _factor(self) -> n.Node:
        expr = self._unary()
        while self._match(T.STAR, T.SLASH, T.PERCENT):
            op_tok = self._previous()
            right = self._unary()
            expr = n.BinaryOp(op=op_tok.lexeme, left=expr, right=right, line=op_tok.line)
        return expr

    def _unary(self) -> n.Node:
        if self._match(T.NOT, T.MINUS):
            op_tok = self._previous()
            operand = self._unary()
            return n.UnaryOp(op=op_tok.lexeme, operand=operand, line=op_tok.line)
        return self._call()

    def _call_args(self) -> tuple[list[n.Node], list[tuple[str, n.Node]]]:
        """`args := (expr | IDENT ":" expr) ("," same)*`, positional
        arguments required before any labeled ones (same order Python's
        own call syntax enforces) -- e.g. `inference("anthropic",
        "claude-sonnet-5", temperature: 0.2)`. Unambiguous with no
        lookahead beyond one token: `:` never otherwise appears at the
        start of an argument (it's only ever a block or type-annotation
        marker elsewhere), so `IDENT ":"` is always a label, never the
        start of some other expression. Deliberately narrower than
        general keyword arguments -- see ast_nodes.py's `Call.kwargs`."""
        args: list[n.Node] = []
        kwargs: list[tuple[str, n.Node]] = []
        if self._check(T.RPAREN):
            return args, kwargs
        while True:
            if self._check(T.IDENT) and self._peek(1).type == T.COLON:
                label = self._advance().lexeme
                self._advance()  # consume ':'
                kwargs.append((label, self._expr()))
            elif kwargs:
                raise SenseSyntaxError(
                    "positional arguments must come before labeled ('name: value') ones", self._peek().line
                )
            else:
                args.append(self._expr())
            if not self._match(T.COMMA):
                break
        return args, kwargs

    def _call(self) -> n.Node:
        expr = self._primary()
        while True:
            if self._match(T.LPAREN):
                line = self._previous().line
                args, kwargs = self._call_args()
                self._expect(T.RPAREN, "expected ')' after the arguments")
                expr = n.Call(callee=expr, args=args, kwargs=kwargs, line=line)
            elif self._match(T.DOT):
                name = self._expect(T.IDENT, "expected a property name after '.'").lexeme
                expr = n.MemberAccess(target=expr, name=name, line=self._previous().line)
            elif self._match(T.LBRACKET):
                line = self._previous().line
                index = self._expr()
                self._expect(T.RBRACKET, "expected ']' after the index")
                expr = n.Index(target=expr, index=index, line=line)
            else:
                break
        return expr

    def _primary(self) -> n.Node:
        tok = self._peek()
        if self._match(T.INT):
            return n.IntLit(value=self._previous().literal, line=tok.line)
        if self._match(T.FLOAT):
            return n.FloatLit(value=self._previous().literal, line=tok.line)
        if self._match(T.STRING):
            return n.StringLit(value=self._previous().literal, line=tok.line)
        if self._match(T.TRUE):
            return n.BoolLit(value=True, line=tok.line)
        if self._match(T.FALSE):
            return n.BoolLit(value=False, line=tok.line)
        if self._match(T.NIL):
            return n.NilLit(line=tok.line)
        if self._match(T.IDENT):
            return n.Identifier(name=self._previous().lexeme, line=tok.line)
        if self._match(T.LPAREN):
            expr = self._expr()
            self._expect(T.RPAREN, "expected ')' after the expression")
            return expr
        if self._match(T.LBRACKET):
            elements = []
            if not self._check(T.RBRACKET):
                elements.append(self._expr())
                while self._match(T.COMMA):
                    elements.append(self._expr())
            self._expect(T.RBRACKET, "expected ']' after the array elements")
            return n.ArrayLit(elements=elements, line=tok.line)

        raise SenseSyntaxError(f"unexpected token {tok.type.name} {tok.lexeme!r}", tok.line)


def parse(tokens: list[Token]) -> n.Program:
    return Parser(tokens).parse_program()
