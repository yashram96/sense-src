"""Sense — a programming language for probabilistic and autonomous computation.

A tree-walking interpreter over a deterministic core (lexer, parser, AST,
a runtime type checker) plus the AI-native primitives built on top of it:
model/ask/Answer, tool, memory, agent (with pause/resume), and the
reversible/irreversible action lifecycle (prepare -> verify -> commit)
with capability policy and human approval.
"""

__version__ = "0.20.0"
