"""Error types shared across the lexer, parser, checker, and interpreter."""

from __future__ import annotations


class SenseError(Exception):
    """Base class for all Sense compile/runtime errors."""

    def __init__(self, message: str, line: int | None = None):
        self.message = message
        self.line = line
        located = f"[line {line}] {message}" if line is not None else message
        super().__init__(located)


class SenseSyntaxError(SenseError):
    """Raised by the lexer or parser on malformed source."""


class SenseNameError(SenseError):
    """Raised when an identifier is not defined in the current scope."""


class SenseTypeError(SenseError):
    """Raised when a value does not conform to a declared type annotation."""


class SenseRuntimeError(SenseError):
    """Raised for runtime failures that aren't type or name errors."""


class SenseImportError(SenseError):
    """Raised for module resolution / circular import failures."""


class SensePolicyError(SenseError):
    """Raised when an Action's required capability is denied by policy."""


class SenseApprovalError(SenseError):
    """Raised when an Action declared 'requires approval' is committed
    before <action>.approve() has been called."""
