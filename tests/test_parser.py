import pytest

from sense_lang import ast_nodes as n
from sense_lang.errors import SenseSyntaxError
from sense_lang.lexer import tokenize
from sense_lang.parser import parse


def parse_src(source: str) -> n.Program:
    return parse(tokenize(source))


def test_typed_decl():
    prog = parse_src("x: Int = 5\n")
    stmt = prog.statements[0]
    assert isinstance(stmt, n.TypedDecl)
    assert stmt.name == "x"
    assert stmt.type_ann.name == "Int"
    assert isinstance(stmt.value, n.IntLit)
    assert stmt.value.value == 5


def test_untyped_assignment_is_expr_stmt_wrapping_assign():
    prog = parse_src("x = 5\n")
    stmt = prog.statements[0]
    assert isinstance(stmt, n.ExprStmt)
    assert isinstance(stmt.expr, n.Assign)
    assert stmt.expr.name == "x"


def test_fn_decl_requires_def_and_arrow_return_type():
    prog = parse_src("def add(a, b) -> Int:\n    return a + b\n")
    fn = prog.statements[0]
    assert isinstance(fn, n.FnDecl)
    assert fn.name == "add"
    assert [p.name for p in fn.params] == ["a", "b"]
    assert fn.return_type.name == "Int"
    ret = fn.body.statements[0]
    assert isinstance(ret, n.ReturnStmt)
    assert isinstance(ret.value, n.BinaryOp)


def test_fn_decl_with_typed_params_no_return_type():
    prog = parse_src("def square(x: Int):\n    return x * x\n")
    fn = prog.statements[0]
    assert fn.params[0].type_ann.name == "Int"
    assert fn.return_type is None


def test_fn_decl_without_def_raises_a_helpful_error():
    with pytest.raises(SenseSyntaxError, match="'def'"):
        parse_src("add(a, b) returns Int:\n    return a + b\n")


def test_call_is_not_confused_with_fn_decl():
    prog = parse_src("foo(1, 2)\n")
    stmt = prog.statements[0]
    assert isinstance(stmt, n.ExprStmt)
    assert isinstance(stmt.expr, n.Call)


def test_generic_array_type_annotation():
    prog = parse_src('xs: Array<String> = ["a", "b"]\n')
    ann = prog.statements[0].type_ann
    assert ann.name == "Array"
    assert ann.params[0].name == "String"


def test_operator_precedence():
    prog = parse_src("x = 1 + 2 * 3\n")
    value = prog.statements[0].expr.value
    assert isinstance(value, n.BinaryOp)
    assert value.op == "+"
    assert isinstance(value.right, n.BinaryOp)
    assert value.right.op == "*"


def test_word_logical_operators():
    prog = parse_src("x = a and b or not c\n")
    value = prog.statements[0].expr.value
    assert isinstance(value, n.LogicalOp)
    assert value.op == "or"
    assert isinstance(value.left, n.LogicalOp)
    assert value.left.op == "and"


def test_if_else_if_chain_indentation_based():
    prog = parse_src("if a:\n    x = 1\nelse if b:\n    x = 2\nelse:\n    x = 3\n")
    stmt = prog.statements[0]
    assert isinstance(stmt, n.IfStmt)
    assert isinstance(stmt.else_branch, n.IfStmt)
    assert isinstance(stmt.else_branch.else_branch, n.Block)


def test_inline_single_statement_block():
    prog = parse_src("if a: return 1\n")
    stmt = prog.statements[0]
    assert isinstance(stmt, n.IfStmt)
    assert len(stmt.then_branch.statements) == 1
    assert isinstance(stmt.then_branch.statements[0], n.ReturnStmt)


def test_call_index_and_member_chain():
    prog = parse_src("a.b(1, 2)[0]\n")
    expr = prog.statements[0].expr
    assert isinstance(expr, n.Index)
    assert isinstance(expr.target, n.Call)
    assert isinstance(expr.target.callee, n.MemberAccess)


def test_assignment_and_index_assignment():
    prog = parse_src("x = 1\narr[0] = 2\n")
    assert isinstance(prog.statements[0].expr, n.Assign)
    assert isinstance(prog.statements[1].expr, n.IndexAssign)


def test_import_with_alias():
    prog = parse_src('import "./m.sns" as m\n')
    stmt = prog.statements[0]
    assert isinstance(stmt, n.ImportStmt)
    assert stmt.path == "./m.sns"
    assert stmt.alias == "m"


def test_for_in_loop():
    prog = parse_src("for item in items:\n    x = item\n")
    stmt = prog.statements[0]
    assert isinstance(stmt, n.ForStmt)
    assert stmt.var_name == "item"


def test_set_stmt():
    prog = parse_src("set delegation = m\n")
    stmt = prog.statements[0]
    assert isinstance(stmt, n.SetStmt)
    assert stmt.name == "delegation"


def test_missing_paren_raises_syntax_error():
    with pytest.raises(SenseSyntaxError):
        parse_src("f(a, b\n    return 1\n")


def test_invalid_assignment_target_raises():
    with pytest.raises(SenseSyntaxError):
        parse_src("1 = 2\n")


def test_missing_indented_block_raises():
    with pytest.raises(SenseSyntaxError):
        parse_src("if true:\nx = 1\n")
