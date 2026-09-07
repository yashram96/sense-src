"""sense fmt's pretty-printer.

Core guarantees under test: formatting never changes what a program does
(verified against every real example, not just synthetic snippets), and
formatting is idempotent (fmt(fmt(x)) == fmt(x)).
"""

from pathlib import Path
from textwrap import dedent

import pytest

from sense_lang.formatter import format_source
from sense_lang.interpreter import Interpreter

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLE_FILES = sorted(EXAMPLES_DIR.rglob("*.sns"))

# Deliberately ends in an uncaught SenseError -- see its own header
# comment and tests/test_examples.py. Formatting it is still checked
# (test_formatting_is_idempotent, test_example_corpus_is_already_formatted
# below both just transform text, never run it); only the "run it and
# compare output" test needs to skip it.
_RUNS_AND_FAILS_ON_PURPOSE = "rollback_approval_workflow_failure.sns"


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(source)
    return output


def fmt(source: str) -> str:
    return format_source(dedent(source).lstrip("\n"))


# -- the two guarantees, checked against every real example ---------------------


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: str(p.relative_to(EXAMPLES_DIR)))
def test_formatting_is_idempotent(path: Path):
    source = path.read_text(encoding="utf-8")
    once = format_source(source)
    twice = format_source(once)
    assert once == twice


@pytest.mark.parametrize("path", EXAMPLE_FILES, ids=lambda p: str(p.relative_to(EXAMPLES_DIR)))
def test_formatting_preserves_behavior(path: Path):
    if path.name == _RUNS_AND_FAILS_ON_PURPOSE:
        pytest.skip("deliberately ends in an uncaught error -- see test_examples.py for its own test")
    source = path.read_text(encoding="utf-8")
    formatted = format_source(source)
    # run_source (not run_file) for both -- imports resolve relative to CWD
    # either way, so any import-path difference is neutralized; what's
    # under test is "does reformatted *content* behave the same," not
    # module resolution.
    if "import " in source:
        pytest.skip("relative imports need run_file with a real file path, not run_source")
    original_output = run(source)
    formatted_output = run(formatted)
    assert original_output == formatted_output


def test_example_corpus_is_already_formatted():
    """Guards against the example corpus drifting from the formatter's own
    canonical style over time (see the .sns files under examples/, which
    were brought in line with `sense fmt --write` when this test was
    added)."""
    for path in EXAMPLE_FILES:
        source = path.read_text(encoding="utf-8")
        assert format_source(source) == source, f"{path} is not canonically formatted"


# -- expression precedence: minimal parens, both directions ---------------------


def test_no_unnecessary_parens_removed_by_reprinting():
    assert fmt("x = 1 + 2 * 3\n") == "x = 1 + 2 * 3\n"


def test_necessary_parens_are_preserved():
    assert fmt("x = (1 + 2) * 3\n") == "x = (1 + 2) * 3\n"


def test_left_associative_chain_needs_no_parens():
    assert fmt("x = 1 - 2 - 3\n") == "x = 1 - 2 - 3\n"


def test_right_grouped_same_precedence_needs_parens():
    # 1 - (2 - 3) is NOT the same value as 1 - 2 - 3, so the parens are semantic
    assert fmt("x = 1 - (2 - 3)\n") == "x = 1 - (2 - 3)\n"


def test_and_or_precedence_parens():
    assert fmt("x = a and b or c\n") == "x = a and b or c\n"
    assert fmt("x = a and (b or c)\n") == "x = a and (b or c)\n"


def test_unary_minus_of_unary_minus_gets_a_space():
    assert fmt("x = - -5\n") == "x = - -5\n"


def test_call_on_grouped_expression_keeps_parens():
    assert fmt("x = (a + b)(1)\n") == "x = (a + b)(1)\n"


def test_redundant_grouping_parens_are_dropped():
    assert fmt("x = (5)\n") == "x = 5\n"
    assert fmt("x = (a)\n") == "x = a\n"


# -- statement canonicalization ---------------------------------------------------


def test_single_rule_policy_collapses_to_inline():
    assert fmt("policy:\n    allow x.y\n") == "policy: allow x.y\n"


def test_multi_rule_policy_stays_multiline():
    out = fmt("policy:\n    allow x.y\n    deny a.b\n")
    assert out == "policy:\n    allow x.y\n    deny a.b\n"


def test_inline_policy_with_multiple_rules_expands():
    # can't actually have >1 rule in inline form syntactically; single stays inline
    assert fmt("policy: allow x.y\n") == "policy: allow x.y\n"


def test_action_requires_canonical_order_capability_then_approval():
    src = "irreversible action f() requires approval, x.y:\n    print(1)\n"
    out = fmt(src)
    assert out == "irreversible action f() requires x.y, approval:\n    print(1)\n"


def test_action_requires_capability_only():
    src = "irreversible action f() requires x.y:\n    print(1)\n"
    assert fmt(src) == src


def test_action_requires_approval_only():
    src = "irreversible action f() requires approval:\n    print(1)\n"
    assert fmt(src) == src


def test_inline_if_block_expands_to_multiline():
    out = fmt("if x: return 1\n")
    assert out == "if x:\n    return 1\n"


def test_typed_decl_roundtrip():
    assert fmt("x: Int = 5\n") == "x: Int = 5\n"


def test_local_decl_roundtrip():
    assert fmt("local x: Int = 5\n") == "local x: Int = 5\n"
    assert fmt("local x = 5\n") == "local x = 5\n"


def test_function_with_typed_params_and_return():
    src = "def add(a: Int, b: Int) -> Int:\n    return a + b\n"
    assert fmt(src) == src


def test_if_elif_else_chain():
    src = "if a:\n    x = 1\nelse if b:\n    x = 2\nelse:\n    x = 3\n"
    assert fmt(src) == src


def test_for_loop_roundtrip():
    src = "for x in xs:\n    print(x)\n"
    assert fmt(src) == src


def test_import_roundtrip():
    assert fmt('import "./m.sns" as m\n') == 'import "./m.sns" as m\n'
    assert fmt('import "./m.sns"\n') == 'import "./m.sns"\n'


def test_import_python_keyword_is_preserved():
    # Regression: dropping 'python' here would silently change the
    # statement's meaning (a Sense file import vs. a Python module import),
    # violating the "formatting never changes behavior" guarantee.
    assert fmt('import python "math" as math\n') == 'import python "math" as math\n'
    assert fmt('import python "os.path"\n') == 'import python "os.path"\n'


def test_array_literal_roundtrip():
    assert fmt("xs = [1, 2, 3]\n") == "xs = [1, 2, 3]\n"


def test_string_escaping_roundtrip():
    """A one-line value (no embedded newline) still round-trips through
    escape sequences exactly as before."""
    src = 'x = "a\\tb\\"c"\n'
    assert fmt(src) == src


def test_multiline_string_prints_as_triple_quoted_not_escaped_one_liner():
    """A string whose *value* contains a real newline -- however it was
    written (an actual multi-line "..." pair, which already tolerated
    embedded newlines before triple-quote support existed, or a `\\n`
    escape sequence) -- now prints back out as an actual multi-line
    `\"\"\"...\"\"\"` literal instead of one very long line with a literal
    `\\n` in it. This is a deliberate change from the escaped-one-liner
    form the (now removed) old version of this test asserted: that
    behavior predates multi-line strings existing at all, and printing a
    docstring -- the whole reason multi-line strings exist -- back out as
    an escaped one-liner would defeat the point. Both formatter
    guarantees (idempotent, behavior-preserving) still hold either way,
    verified directly here rather than assumed."""
    src = 'x = "a\\nb\\tc\\"d"\n'
    formatted = fmt(src)
    assert formatted == 'x = """a\nb\tc"d"""\n'
    assert fmt(formatted) == formatted  # idempotent
    assert run(src) == run(formatted)  # behavior-preserving


def test_test_statement_roundtrip():
    src = 'test "it works":\n    assert(1 == 1)\n'
    assert fmt(src) == src


def test_tool_decl_roundtrip():
    src = 'tool search(query: String) returns String requires web.search:\n    return query\n'
    assert fmt(src) == src
    assert fmt("tool search(query: String):\n    return query\n") == "tool search(query: String):\n    return query\n"


def test_tool_description_is_preserved():
    # Regression risk mirrors 'import python'/'rollback'/'async': dropping
    # the description on reformat would silently make the tool unusable
    # with ask_with_tools() (it requires one) without any parse error.
    src = 'tool search(query: String) returns String requires web.search "search the web":\n    return query\n'
    assert fmt(src) == src
    assert fmt('tool search(query: String) "search the web":\n    return query\n') == 'tool search(query: String) "search the web":\n    return query\n'


def test_memory_decl_roundtrip():
    assert fmt('memory notes: "working"\n') == 'memory notes: "working"\n'
    assert fmt("memory notes\n") == "memory notes\n"


def test_persistent_memory_keyword_is_preserved():
    # Regression risk mirrors the 'import python' bug: dropping
    # 'persistent' here would silently turn a durable memory into an
    # in-process one that vanishes when the interpreter exits.
    assert fmt('persistent memory ledger: "data/notes.db"\n') == 'persistent memory ledger: "data/notes.db"\n'
    assert fmt("persistent memory ledger\n") == "persistent memory ledger\n"


def test_rollback_block_is_preserved():
    # Regression risk mirrors the 'import python' bug: dropping the
    # rollback block on reformat would silently remove an action's
    # rollback path without any error.
    src = (
        "reversible action update_profile(user: String) requires profile.write:\n"
        "    apply(user)\n"
        "rollback:\n"
        "    revert(user)\n"
    )
    assert fmt(src) == src


def test_async_agent_keyword_is_preserved():
    # Regression risk mirrors the 'import python' bug: dropping 'async'
    # here would silently turn a non-blocking agent into a blocking one.
    src = 'async agent a:\n    x = 1\n'
    assert fmt(src) == src
    assert fmt("agent a:\n    x = 1\n") == "agent a:\n    x = 1\n"


def test_async_tool_keyword_is_preserved():
    src = "async tool search(query: String) returns String:\n    return query\n"
    assert fmt(src) == src
    assert fmt("tool search(query: String):\n    return query\n") == "tool search(query: String):\n    return query\n"


# -- comments -----------------------------------------------------------------


def test_standalone_comment_preserved_before_statement():
    out = fmt("# a comment\nx = 5\n")
    assert out == "# a comment\nx = 5\n"


def test_trailing_comment_stays_on_same_line():
    out = fmt("x = 5  # note\n")
    assert out == "x = 5  # note\n"


def test_trailing_comment_spacing_is_normalized_to_two_spaces():
    out = fmt("x = 5     # note\n")
    assert out == "x = 5  # note\n"


def test_comment_text_is_never_dropped():
    src = "# alpha\n# beta\nx = 1\n# gamma\n"
    out = fmt(src)
    for word in ("alpha", "beta", "gamma"):
        assert word in out


def test_blank_line_after_comment_is_preserved():
    out = fmt("# header\n\nx = 5\n")
    assert out == "# header\n\nx = 5\n"


def test_no_blank_line_added_when_source_had_none():
    out = fmt("# header\nx = 5\n")
    assert out == "# header\nx = 5\n"


def test_multiple_blank_lines_collapse_to_one():
    out = fmt("x = 1\n\n\n\ny = 2\n")
    assert out == "x = 1\n\ny = 2\n"


def test_comment_only_program():
    assert fmt("# just a comment\n") == "# just a comment\n"


def test_empty_program_formats_to_empty_string():
    assert format_source("") == ""


# -- idempotency on hand-built inputs (in addition to the real-example sweep) ----


@pytest.mark.parametrize(
    "source",
    [
        "x = 1 + 2 * 3\n",
        "policy:\n    allow x.y\n    deny a.b\n",
        "if a:\n    x = 1\nelse if b:\n    x = 2\nelse:\n    x = 3\n",
        "# c\nx = 5  # d\n\ny = 6\n",
    ],
)
def test_idempotent_on_hand_built_snippets(source):
    once = format_source(source)
    twice = format_source(once)
    assert once == twice
