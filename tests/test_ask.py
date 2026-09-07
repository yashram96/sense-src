"""Phase 2: model / set delegation / ask / answer.

`model NAME = inference(...)` is required to name a `Model` -- a plain
`NAME = inference(...)` is a parse-time error (see
test_inference_requires_model_keyword below). `inference(...)` still
works unnamed anywhere else a value is used -- as a `set delegation = ...`
target, as a bare expression, chained straight off the call (`inference
("mock").name`) -- it's only direct assignment to a plain identifier
that's restricted.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseRuntimeError, SenseSyntaxError, SenseTypeError
from sense_lang.interpreter import Interpreter
from sense_lang.values import SenseModel


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


def test_inference_mock_returns_a_model_value():
    out = run('model m = inference("mock", "x")\nprint(type_of(m))\n')
    assert out == ["Model"]


def test_ask_without_delegation_raises():
    with pytest.raises(SenseRuntimeError):
        run('print(ask("hi"))\n')


def test_set_delegation_then_bare_ask():
    out = run(
        """
        model m = inference("mock", "x", response_template: "answer: {prompt}")
        set delegation = m
        r = ask("q")
        print(r.value)
        print(type_of(r))
        """
    )
    assert out == ["answer: q", "Answer"]


def test_set_delegation_rejects_non_model():
    with pytest.raises(SenseTypeError):
        run("set delegation = 5\n")


def test_set_unknown_target_raises():
    with pytest.raises(SenseRuntimeError):
        run("set some_other_thing = 5\n")


def test_explicit_model_dot_ask_ignores_delegation():
    out = run(
        """
        model a = inference("mock", "a", response_template: "from-a: {prompt}")
        model b = inference("mock", "b", response_template: "from-b: {prompt}")
        set delegation = a
        r = b.ask("q")
        print(r.value)
        print(r.source)
        """
    )
    assert out == ["from-b: q", "b"]


def test_answer_member_access():
    out = run(
        """
        model m = inference("mock", "src", response_template: "hello {prompt}")
        set delegation = m
        r = ask("world")
        print(r.value)
        print(r.confidence)
        print(r.source)
        """
    )
    assert out == ["hello world", "0.5", "m"]  # "model m = ..." names it "m", overriding "src"


def test_answer_type_annotation_accepts_matching_answer():
    out = run(
        """
        model m = inference("mock", "src")
        set delegation = m
        r: answer<String> = ask("q")
        print(type_of(r))
        """
    )
    assert out == ["Answer"]


def test_ask_requires_string_prompt():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            model m = inference("mock", "src")
            set delegation = m
            ask(42)
            """
        )


def test_answer_stringify_includes_confidence_and_value():
    out = run(
        """
        model m = inference("mock", "src", response_template: "x")
        set delegation = m
        print(ask("ignored"))
        """
    )
    assert out == ["answer<0.50>(x)"]


def test_env_current_model_is_a_sense_model_instance():
    interp = Interpreter(stdout=lambda _: None)
    env = interp.run_source(
        dedent(
            """
            model m = inference("mock", "src")
            set delegation = m
            r = ask("q")
            """
        )
    )
    assert isinstance(env.get_current_model(), SenseModel)


def test_inference_mock_defaults_name_to_mock_with_no_second_argument():
    out = run('print(inference("mock").name)\n')
    assert out == ["mock"]


def test_inference_mock_confidence_option_is_configurable():
    out = run(
        """
        model m = inference("mock", "src", confidence: 0.9)
        set delegation = m
        print(ask("q").confidence)
        """
    )
    assert out == ["0.9"]


def test_inference_requires_model_keyword():
    with pytest.raises(SenseSyntaxError, match="needs the 'model' keyword"):
        run('x = inference("mock", "demo")\n')


def test_inference_requires_model_keyword_for_real_providers_too():
    with pytest.raises(SenseSyntaxError, match="needs the 'model' keyword"):
        run('x = inference("anthropic")\n')


def test_model_keyword_form_is_unaffected():
    out = run('model x = inference("mock", "demo")\nprint(type_of(x))\n')
    assert out == ["Model"]


def test_set_delegation_is_exempt_from_the_model_keyword_requirement():
    # `set delegation = inference(...)` isn't naming a variable -- it's
    # activating delegation, the same as any other `set` target -- so it
    # isn't required to go through `model`.
    out = run('set delegation = inference("mock", "demo")\nprint(ask("hi").value)\n')
    assert out == ["(mock reasoning about: hi)"]


def test_bare_expression_statement_is_exempt_from_the_model_keyword_requirement():
    # Not an assignment at all -- nothing to require `model` for.
    out = run('inference("mock", "demo")\n')
    assert out == []


def test_inference_mock_rejects_real_provider_only_options():
    with pytest.raises(SenseRuntimeError):
        run('inference("mock", "src", temperature: 0.2)\n')


def test_inference_real_provider_rejects_mock_only_options():
    with pytest.raises(SenseRuntimeError):
        run('inference("anthropic", "claude-sonnet-5", response_template: "x")\n')


def test_mock_model_is_no_longer_a_valid_call():
    with pytest.raises(SenseRuntimeError):
        run('mock_model("x")\n')
