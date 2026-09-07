"""Hand-written scanner: Sense source text -> list[Token].

Blocks are indentation, not braces, so this scanner does what Python's
does: track leading whitespace per logical line and synthesize
INDENT/DEDENT tokens, plus a NEWLINE token at the end of each
non-blank logical line. Newlines and indentation are both ignored while
inside `( )` / `[ ]` (tracked via `paren_depth`), so a call or array literal
may freely wrap across lines.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SenseSyntaxError
from .tokens import KEYWORDS, Token, TokenType


@dataclass(frozen=True)
class Comment:
    line: int
    text: str
    trailing: bool  # True if it followed real code on the same line


_SIMPLE_TOKENS = {
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
    ".": TokenType.DOT,
    "+": TokenType.PLUS,
    "*": TokenType.STAR,
    "%": TokenType.PERCENT,
}

_OPENERS = "(["
_CLOSERS = ")]"


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.start = 0
        self.pos = 0
        self.line = 1
        self.tokens: list[Token] = []
        self.paren_depth = 0
        self.indent_stack: list[int] = [0]
        self.at_line_start = True
        self.line_has_content = False
        # One Comment per '#' comment, '#' and surrounding whitespace
        # stripped. Not part of the token stream -- the parser never sees
        # these -- but `sense fmt` (formatter.py) needs them to avoid
        # silently deleting comments, since they aren't part of the AST.
        self.comments: list[Comment] = []

    def tokenize(self) -> list[Token]:
        while not self._at_end():
            if self.at_line_start and self.paren_depth == 0:
                self._handle_indentation()
                if self._at_end():
                    break
            self.start = self.pos
            self._scan_token()

        if self.line_has_content:
            self.tokens.append(Token(TokenType.NEWLINE, "", None, self.line))
            self.line_has_content = False
        while len(self.indent_stack) > 1:
            self.indent_stack.pop()
            self.tokens.append(Token(TokenType.DEDENT, "", None, self.line))
        self.tokens.append(Token(TokenType.EOF, "", None, self.line))
        return self.tokens

    # -- helpers ---------------------------------------------------------

    def _at_end(self) -> bool:
        return self.pos >= len(self.source)

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        return ch

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        return self.source[idx] if idx < len(self.source) else "\0"

    def _match(self, expected: str) -> bool:
        if self._at_end() or self.source[self.pos] != expected:
            return False
        self.pos += 1
        return True

    def _add(self, type_: TokenType, literal: object = None) -> None:
        lexeme = self.source[self.start:self.pos]
        self.tokens.append(Token(type_, lexeme, literal, self.line))
        self.line_has_content = True

    # -- indentation -------------------------------------------------------

    def _handle_indentation(self) -> None:
        """Consume blank/comment-only lines, then emit INDENT/DEDENT for the
        next real line's leading whitespace relative to `indent_stack`."""
        while True:
            count = 0
            while self._peek() == " ":
                count += 1
                self.pos += 1
            if self._peek() == "\t":
                raise SenseSyntaxError(
                    "tabs are not allowed for indentation — use spaces", self.line
                )
            if self._peek() == "\n":
                self.pos += 1
                self.line += 1
                continue
            if self._peek() == "#":
                comment_start = self.pos
                while self._peek() != "\n" and not self._at_end():
                    self.pos += 1
                text = self.source[comment_start:self.pos][1:].strip()
                self.comments.append(Comment(self.line, text, trailing=False))
                if self._at_end():
                    break
                self.pos += 1
                self.line += 1
                continue
            break

        if self._at_end():
            self.at_line_start = False
            return

        top = self.indent_stack[-1]
        if count > top:
            self.indent_stack.append(count)
            self.tokens.append(Token(TokenType.INDENT, "", None, self.line))
        elif count < top:
            while count < self.indent_stack[-1]:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, "", None, self.line))
            if count != self.indent_stack[-1]:
                raise SenseSyntaxError("inconsistent indentation", self.line)
        self.at_line_start = False

    # -- scanning ----------------------------------------------------------

    def _scan_token(self) -> None:
        ch = self._advance()

        if ch in " \r\t":
            return
        if ch == "\n":
            self.line += 1
            if self.paren_depth == 0:
                if self.line_has_content:
                    self.tokens.append(Token(TokenType.NEWLINE, "", None, self.line - 1))
                    self.line_has_content = False
                self.at_line_start = True
            return
        if ch == "#":
            while self._peek() != "\n" and not self._at_end():
                self._advance()
            text = self.source[self.start:self.pos][1:].strip()
            self.comments.append(Comment(self.line, text, trailing=self.line_has_content))
            return

        if ch in _SIMPLE_TOKENS:
            self._add(_SIMPLE_TOKENS[ch])
            if ch in _OPENERS:
                self.paren_depth += 1
            elif ch in _CLOSERS:
                self.paren_depth = max(0, self.paren_depth - 1)
            return

        if ch == "{" or ch == "}":
            raise SenseSyntaxError(
                "Sense uses indentation for blocks, not '{ }' — write ':' then an "
                "indented block instead",
                self.line,
            )

        if ch == "/":
            self._add(TokenType.SLASH)
            return

        if ch == "=":
            self._add(TokenType.EQEQ if self._match("=") else TokenType.EQ)
            return
        if ch == "!":
            if self._match("="):
                self._add(TokenType.BANGEQ)
            else:
                raise SenseSyntaxError(
                    "'!' is not an operator in Sense — use 'not' for negation and "
                    "'!=' for not-equal",
                    self.line,
                )
            return
        if ch == "<":
            self._add(TokenType.LTEQ if self._match("=") else TokenType.LT)
            return
        if ch == ">":
            self._add(TokenType.GTEQ if self._match("=") else TokenType.GT)
            return
        if ch == "-":
            if self._match(">"):
                self._add(TokenType.ARROW)
            else:
                self._add(TokenType.MINUS)
            return
        if ch == ";":
            raise SenseSyntaxError(
                "semicolons aren't used in Sense — start a new line instead", self.line
            )
        if ch == '"':
            self._string()
            return

        if ch.isdigit():
            self._number()
            return
        if ch.isalpha() or ch == "_":
            self._identifier()
            return

        raise SenseSyntaxError(f"unexpected character {ch!r}", self.line)

    def _string(self) -> None:
        # A plain "..." pair already tolerates an embedded literal newline
        # (the loop below just keeps consuming characters, bumping
        # self.line as it goes) -- so multi-line strings work today with
        # no triple-quote syntax at all. `"""` is still worth recognizing
        # on its own: without it, `"""text"""` lexes as three back-to-back
        # STRING tokens (`""`, `"text"`, `""`) with nothing joining them,
        # which is exactly the confusing parse error a Python-habituated
        # docstring attempt hits. Triple-quoted mode only changes what
        # ends the string (three quotes instead of one) and allows a bare
        # `"` inside unescaped -- same escape handling otherwise.
        if self._peek() == '"' and self._peek(1) == '"':
            self._advance()
            self._advance()
            self._triple_quoted_string()
            return
        value_chars: list[str] = []
        while self._peek() != '"' and not self._at_end():
            c = self._advance()
            if c == "\n":
                self.line += 1
            if c == "\\":
                esc = self._advance()
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}
                value_chars.append(mapping.get(esc, esc))
            else:
                value_chars.append(c)
        if self._at_end():
            raise SenseSyntaxError("unterminated string literal", self.line)
        self._advance()  # closing quote
        self._add(TokenType.STRING, "".join(value_chars))

    def _triple_quoted_string(self) -> None:
        value_chars: list[str] = []
        while not self._at_end() and not (
            self._peek() == '"' and self._peek(1) == '"' and self._peek(2) == '"'
        ):
            c = self._advance()
            if c == "\n":
                self.line += 1
            if c == "\\":
                esc = self._advance()
                mapping = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "r": "\r"}
                value_chars.append(mapping.get(esc, esc))
            else:
                value_chars.append(c)
        if self._at_end():
            raise SenseSyntaxError("unterminated triple-quoted string literal", self.line)
        self._advance()
        self._advance()
        self._advance()  # closing \"\"\"
        self._add(TokenType.STRING, "".join(value_chars))

    def _number(self) -> None:
        while self._peek().isdigit():
            self._advance()
        is_float = False
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()
            while self._peek().isdigit():
                self._advance()
        text = self.source[self.start:self.pos]
        if is_float:
            self._add(TokenType.FLOAT, float(text))
        else:
            self._add(TokenType.INT, int(text))

    def _identifier(self) -> None:
        while self._peek().isalnum() or self._peek() == "_":
            self._advance()
        text = self.source[self.start:self.pos]
        type_ = KEYWORDS.get(text, TokenType.IDENT)
        self._add(type_)


def tokenize(source: str) -> list[Token]:
    return Lexer(source).tokenize()


def tokenize_with_comments(source: str) -> tuple[list[Token], list[Comment]]:
    """Like `tokenize`, but also returns the `# ...` comments the scanner
    otherwise discards — used only by `sense fmt` (formatter.py)."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    return tokens, lexer.comments
