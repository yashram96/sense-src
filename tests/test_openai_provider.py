"""A second real model provider (`inference("openai", ...)`) behind the same
`ModelProvider` seam `MockModelProvider`/`AnthropicProvider` already use.
Every test here runs fully offline -- no real network call, no API key
needed -- proving the capability-gating/error-wrapping/audit logic
without ever touching the actual OpenAI API. Mirrors
tests/test_anthropic_provider.py structurally on purpose: same seam,
same guarantees, different vendor -- inference("anthropic", ...) and
inference("openai", ...) are the same function.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SensePolicyError, SenseRuntimeError
from sense_lang.interpreter import Interpreter
from sense_lang.providers import ModelProvider, ModelResponse
from sense_lang.values import SenseModel


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


# -- a fake gated provider, standing in for OpenAIProvider -----------------------
# Proves Interpreter._ask_model's policy-gating logic without needing
# the real `openai` package or a real API key.


class FakeGatedProvider(ModelProvider):
    capability = "model.openai"

    def __init__(self, response_text: str = "real answer"):
        self.response_text = response_text
        self.called = False

    def complete(self, prompt: str) -> ModelResponse:
        self.called = True
        return ModelResponse(text=self.response_text, confidence=None)


class FakeFailingProvider(ModelProvider):
    capability = "model.openai"

    def complete(self, prompt: str) -> ModelResponse:
        raise ConnectionError("simulated network failure")


def make_interpreter(model_name: str, provider: ModelProvider) -> tuple[Interpreter, list[str]]:
    out: list[str] = []
    interp = Interpreter(stdout=out.append)
    interp.globals.define(model_name, SenseModel(name=model_name, provider=provider))
    return interp, out


# -- inference("openai", ...): configuration ---------------------------------------


def test_inference_openai_without_api_key_raises_immediately():
    with pytest.raises(SenseRuntimeError, match="OPENAI_API_KEY"):
        run('inference("openai")\n')


def test_inference_openai_respects_custom_env_var_name():
    with pytest.raises(SenseRuntimeError, match="MY_CUSTOM_KEY"):
        run('inference("openai", "gpt-4o", api_key_env: "MY_CUSTOM_KEY")\n')


def test_inference_openai_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    out = run('model m = inference("openai")\nprint(type_of(m))\n')
    assert out == ["Model"]


def test_inference_openai_default_model_id_is_used_as_the_name(monkeypatch):
    # No `model` keyword here on purpose: this is testing what .name is
    # before anything ever names it, which needs a path that isn't a plain
    # `x = inference(...)` assignment (illegal now -- see
    # test_inference_requires_model_keyword in test_ask.py) -- a direct
    # `.name` chain off the call expression itself is one, and never
    # touches Assign at all, so it's unaffected by that restriction.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    out = run('print(inference("openai").name)\n')
    assert out == ["gpt-4o"]


def test_model_keyword_overrides_the_default_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-for-test")
    out = run('model gpt = inference("openai")\nprint(gpt.name)\n')
    assert out == ["gpt"]


# -- capability gating: proven with a fake provider, no real package needed -------


def test_real_model_call_is_denied_by_policy_before_complete_runs():
    provider = FakeGatedProvider()
    interp, out = make_interpreter("gpt", provider)
    with pytest.raises(SensePolicyError):
        interp.run_source(
            dedent(
                """
                policy:
                    deny model.openai
                set delegation = gpt
                ask("hi")
                """
            )
        )
    assert provider.called is False


def test_real_model_call_succeeds_when_policy_allows():
    provider = FakeGatedProvider(response_text="real answer")
    interp, out = make_interpreter("gpt", provider)
    interp.run_source(
        dedent(
            """
            policy:
                allow model.openai
            set delegation = gpt
            r = ask("hi")
            print(r.value)
            print(r.confidence)
            """
        )
    )
    assert out == ["real answer", "nil"]
    assert provider.called is True


def test_real_model_call_is_allowed_by_default_when_no_policy_mentions_it():
    provider = FakeGatedProvider()
    interp, out = make_interpreter("gpt", provider)
    interp.run_source(
        dedent(
            """
            set delegation = gpt
            print(ask("hi").value)
            """
        )
    )
    assert out == ["real answer"]


def test_provider_failure_is_wrapped_not_a_raw_traceback():
    provider = FakeFailingProvider()
    interp, out = make_interpreter("gpt", provider)
    with pytest.raises(SenseRuntimeError, match="ConnectionError"):
        interp.run_source(
            dedent(
                """
                policy:
                    allow model.openai
                set delegation = gpt
                ask("hi")
                """
            )
        )


def test_real_model_calls_are_audited():
    provider = FakeGatedProvider()
    interp, out = make_interpreter("gpt", provider)
    interp.run_source(
        dedent(
            """
            policy:
                allow model.openai
            set delegation = gpt
            ask("hi")
            for line in audit_log():
                print(line)
            """
        )
    )
    assert any("gpt" in line and "called" in line for line in out)


# -- exercising OpenAIProvider itself, only if the real package is installed.
# `importorskip` is called *inside* the test, not at module level -- at module
# level it would abort collection of this whole file when the package is
# missing, silently losing every offline test above it, not just this one.


class _FakeFunctionCall:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id_: str, name: str, arguments: str):
        self.id = id_
        self.function = _FakeFunctionCall(name, arguments)


class _FakeMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeOpenAIResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


def test_openai_provider_complete_parses_the_real_sdk_response_shape(monkeypatch):
    openai = pytest.importorskip("openai", reason="openai package not installed")
    from sense_lang.providers import OpenAIProvider

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeOpenAIResponse(_FakeMessage("hello from gpt"))

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.chat = FakeChat()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)

    provider = OpenAIProvider(model_id="gpt-4o", api_key="sk-fake")
    response = provider.complete("what is 2+2?")

    assert response.text == "hello from gpt"
    assert response.confidence is None
    assert captured["api_key"] == "sk-fake"
    assert captured["model"] == "gpt-4o"
    assert captured["messages"] == [{"role": "user", "content": "what is 2+2?"}]


def test_openai_provider_sends_max_completion_tokens_not_max_tokens(monkeypatch):
    # Regression test: newer OpenAI models reject the legacy `max_tokens`
    # wire parameter outright ("Unsupported parameter: 'max_tokens' is not
    # supported with this model. Use 'max_completion_tokens' instead.").
    # `inference(..., max_tokens: ...)` is the one Sense-level label across
    # every provider -- only OpenAIProvider's own translation to the
    # current OpenAI wire format needs to know the real parameter name.
    openai = pytest.importorskip("openai", reason="openai package not installed")
    from sense_lang.providers import OpenAIProvider

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeOpenAIResponse(_FakeMessage("hello from gpt"))

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)

    provider = OpenAIProvider(model_id="gpt-5.5", api_key="sk-fake", max_tokens=300)
    provider.complete("hi")

    assert captured["max_completion_tokens"] == 300
    assert "max_tokens" not in captured


def test_openai_provider_complete_with_tools_parses_a_tool_call_response(monkeypatch):
    openai = pytest.importorskip("openai", reason="openai package not installed")
    from sense_lang.providers import OpenAIProvider, ToolSpec

    class FakeCompletions:
        def create(self, **kwargs):
            tool_call = _FakeToolCall("call_1", "get_weather", '{"city": "Paris"}')
            return _FakeOpenAIResponse(_FakeMessage(None, tool_calls=[tool_call]))

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAIClient:
        def __init__(self, api_key):
            self.chat = FakeChat()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAIClient)

    provider = OpenAIProvider(model_id="gpt-4o", api_key="sk-fake")
    spec = ToolSpec(name="get_weather", description="get the weather", parameters={"type": "object", "properties": {}})
    turn = provider.complete_with_tools([{"role": "user", "content": "weather in Paris?"}], [spec])

    assert turn.text is None
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].name == "get_weather"
    assert turn.tool_calls[0].arguments == {"city": "Paris"}
    # The assistant message must be echoed back in OpenAI's own shape,
    # tool_calls included, on the next turn.
    assert turn.raw_assistant_message["tool_calls"][0]["id"] == "call_1"


def test_openai_provider_format_tool_results_is_one_message_per_call():
    openai = pytest.importorskip("openai", reason="openai package not installed")
    from sense_lang.providers import OpenAIProvider

    provider = OpenAIProvider.__new__(OpenAIProvider)  # skip __init__, no client needed
    messages = provider.format_tool_results([("call_1", "sunny"), ("call_2", "rainy")])
    assert messages == [
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "call_2", "content": "rainy"},
    ]
