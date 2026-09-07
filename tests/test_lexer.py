from sense_lang.errors import SenseSyntaxError
from sense_lang.lexer import tokenize
from sense_lang.tokens import TokenType as T

import pytest


def types_of(source: str) -> list[T]:
    return [t.type for t in tokenize(source)]


def test_empty_source_yields_eof_only():
    assert types_of("") == [T.EOF]


def test_literals():
    toks = tokenize('x = 42\ny = 3.5\ns = "hi"\nb = true\n')
    kinds = [t.type for t in toks]
    assert T.INT in kinds
    assert T.FLOAT in kinds
    assert T.STRING in kinds
    assert T.TRUE in kinds


def test_string_escapes():
    toks = tokenize('"a\\nb\\t\\"c\\""')
    assert toks[0].literal == 'a\nb\t"c"'


def test_plain_string_already_tolerates_an_embedded_literal_newline():
    """No triple-quote needed for this -- the single-quote scanner already
    just keeps consuming characters (bumping the line counter) across a
    real newline, same as it always has."""
    toks = tokenize('"line one\nline two"')
    assert toks[0].type == T.STRING
    assert toks[0].literal == "line one\nline two"


def test_triple_quoted_string_is_one_token_not_three():
    """Before this, \"\"\"text\"\"\" lexed as three back-to-back STRING
    tokens (`""`, `"text"`, `""`) with nothing joining them -- exactly the
    confusing parse error a Python-habituated docstring attempt hits."""
    toks = tokenize('"""hello"""')
    kinds = [t.type for t in toks if t.type != T.NEWLINE and t.type != T.EOF]
    assert kinds == [T.STRING]
    assert toks[0].literal == "hello"


def test_triple_quoted_string_spans_multiple_lines():
    toks = tokenize('"""\nline one\nline two\n"""')
    assert toks[0].type == T.STRING
    assert toks[0].literal == "\nline one\nline two\n"


def test_triple_quoted_string_allows_a_bare_unescaped_quote_inside():
    toks = tokenize('"""she said "hi" to him"""')
    assert toks[0].literal == 'she said "hi" to him'


def test_triple_quoted_string_still_supports_escapes():
    toks = tokenize('"""a\\tb\\\\c"""')
    assert toks[0].literal == "a\tb\\c"


def test_unterminated_triple_quoted_string_raises():
    with pytest.raises(SenseSyntaxError):
        tokenize('"""unterminated')


def test_operators_and_punctuation():
    toks = tokenize("== != <= >= : . , [ ] ( )")
    kinds = [t.type for t in toks if t.type != T.NEWLINE][:-1]
    assert kinds == [
        T.EQEQ, T.BANGEQ, T.LTEQ, T.GTEQ,
        T.COLON, T.DOT, T.COMMA, T.LBRACKET, T.RBRACKET, T.LPAREN, T.RPAREN,
    ]


def test_comments_are_ignored():
    toks = tokenize("# a comment\nx = 1\n")
    kinds = [t.type for t in toks]
    assert kinds.count(T.INT) == 1
    assert T.IDENT in kinds


def test_keywords_vs_identifiers():
    toks = tokenize("if else while for in break continue true false nil import as set and or not returns return foo\n")
    kinds = [t.type for t in toks]
    assert kinds[:-3] == [
        T.IF, T.ELSE, T.WHILE, T.FOR, T.IN, T.BREAK, T.CONTINUE,
        T.TRUE, T.FALSE, T.NIL, T.IMPORT, T.AS, T.SET, T.AND, T.OR, T.NOT,
        T.RETURNS, T.RETURN,
    ]
    assert kinds[-3] == T.IDENT  # 'foo'


def test_unterminated_string_raises():
    with pytest.raises(SenseSyntaxError):
        tokenize('"unterminated')


def test_unknown_character_raises():
    with pytest.raises(SenseSyntaxError):
        tokenize("@")


def test_line_tracking():
    toks = tokenize("a = 1\nb = 2\n")
    b_tok = next(t for t in toks if t.type == T.IDENT and t.lexeme == "b")
    assert b_tok.line == 2


def test_braces_rejected_with_helpful_message():
    with pytest.raises(SenseSyntaxError, match="indentation"):
        tokenize("if true {\n    x = 1\n}\n")


def test_semicolons_rejected_with_helpful_message():
    with pytest.raises(SenseSyntaxError, match="[Ss]emicolon"):
        tokenize("x = 1;\n")


def test_bang_rejected_but_bangeq_allowed():
    with pytest.raises(SenseSyntaxError, match="not"):
        tokenize("!true\n")
    assert types_of("1 != 2\n")[:3] == [T.INT, T.BANGEQ, T.INT]


def test_arrow_lexes_as_its_own_token():
    # '->' is a real token now: it's the return-type arrow for `def`
    # function declarations (see parser.py's `_fn_decl`).
    assert types_of("def f() -> Int:\n    return 1\n")[:6] == [
        T.DEF,
        T.IDENT,
        T.LPAREN,
        T.RPAREN,
        T.ARROW,
        T.IDENT,
    ]


def test_tabs_in_indentation_rejected():
    with pytest.raises(SenseSyntaxError, match="[Tt]ab"):
        tokenize("if true:\n\tprint(1)\n")


def test_indentation_produces_indent_and_dedent():
    toks = tokenize("if true:\n    x = 1\ny = 2\n")
    kinds = [t.type for t in toks]
    assert kinds == [
        T.IF, T.TRUE, T.COLON, T.NEWLINE,
        T.INDENT, T.IDENT, T.EQ, T.INT, T.NEWLINE,
        T.DEDENT, T.IDENT, T.EQ, T.INT, T.NEWLINE,
        T.EOF,
    ]


def test_blank_and_comment_lines_do_not_affect_indentation():
    toks = tokenize("if true:\n    x = 1\n\n    # a comment\n    y = 2\n")
    kinds = [t.type for t in toks]
    # exactly one INDENT (for the block) and one trailing DEDENT at EOF
    assert kinds.count(T.INDENT) == 1
    assert kinds.count(T.DEDENT) == 1


def test_inconsistent_indentation_raises():
    with pytest.raises(SenseSyntaxError, match="indentation"):
        tokenize("if true:\n    x = 1\n  y = 2\n")


def test_newlines_and_indentation_suppressed_inside_parens():
    toks = tokenize("f(\n    1,\n    2,\n)\n")
    kinds = [t.type for t in toks]
    assert kinds.count(T.NEWLINE) == 1  # only the one ending the whole statement
    assert T.INDENT not in kinds
    assert T.DEDENT not in kinds


def test_newlines_suppressed_inside_array_literal():
    toks = tokenize("xs = [\n    1,\n    2,\n]\n")
    kinds = [t.type for t in toks]
    assert kinds.count(T.NEWLINE) == 1  # only the one ending the whole statement
