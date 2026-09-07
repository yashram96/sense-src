"""Token types for the Sense lexer.

Current grammar: indentation-based blocks (no braces), no `let`,
word-based `and`/`or`/`not`. `tool`/`action`/`agent` keep their own
keyword-first shape and `returns` for a return type; a plain function is
the one exception, using `def`/`->` (Python's own shape) instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenType(Enum):
    # Literals
    INT = auto()
    FLOAT = auto()
    STRING = auto()
    IDENT = auto()

    # Keywords
    DEF = auto()
    RETURN = auto()
    RETURNS = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    BREAK = auto()
    CONTINUE = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    PYTHON = auto()
    SET = auto()
    LOCAL = auto()
    SESSION = auto()
    AGENT = auto()
    ACTION = auto()
    REVERSIBLE = auto()
    IRREVERSIBLE = auto()
    REQUIRES = auto()
    APPROVAL = auto()
    POLICY = auto()
    ALLOW = auto()
    DENY = auto()
    TEST = auto()
    TOOL = auto()
    MEMORY = auto()
    ASYNC = auto()
    PERSISTENT = auto()
    MODEL = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    EQ = auto()
    EQEQ = auto()
    BANGEQ = auto()
    LT = auto()
    LTEQ = auto()
    GT = auto()
    GTEQ = auto()
    ARROW = auto()

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    COLON = auto()
    DOT = auto()

    # Layout (produced by the indentation-tracking lexer)
    NEWLINE = auto()
    INDENT = auto()
    DEDENT = auto()

    EOF = auto()


KEYWORDS = {
    "def": TokenType.DEF,
    "return": TokenType.RETURN,
    "returns": TokenType.RETURNS,
    "if": TokenType.IF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "nil": TokenType.NIL,
    "import": TokenType.IMPORT,
    "from": TokenType.FROM,
    "python": TokenType.PYTHON,
    "as": TokenType.AS,
    "set": TokenType.SET,
    "local": TokenType.LOCAL,
    "session": TokenType.SESSION,
    "agent": TokenType.AGENT,
    "action": TokenType.ACTION,
    "reversible": TokenType.REVERSIBLE,
    "irreversible": TokenType.IRREVERSIBLE,
    "requires": TokenType.REQUIRES,
    "approval": TokenType.APPROVAL,
    "policy": TokenType.POLICY,
    "allow": TokenType.ALLOW,
    "deny": TokenType.DENY,
    "test": TokenType.TEST,
    "tool": TokenType.TOOL,
    "memory": TokenType.MEMORY,
    "async": TokenType.ASYNC,
    "persistent": TokenType.PERSISTENT,
    "model": TokenType.MODEL,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
}


@dataclass(frozen=True)
class Token:
    type: TokenType
    lexeme: str
    literal: object
    line: int

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Token({self.type.name}, {self.lexeme!r}, line={self.line})"
