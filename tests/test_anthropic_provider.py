"""A real model provider (`inference("anthropic", ...)`) behind the same
`ModelProvider` seam `MockModelProvider` already uses. Every test here runs
fully offline -- no real network call, no API key needed -- proving the
capability-gating/error-wrapping/audit logic without ever touching the
actual Anthropic API.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SensePolicyError, SenseRuntimeError, SenseTypeError
from sense_lang.interpreter import Interpreter
from sense_lang.providers import ModelProvider, ModelResponse
from sense_lang.values import SenseModel


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


# -- a fake gated provider, standing in for AnthropicProvider ---------------------
# Proves Interpreter._ask_model's policy-gating logic without needing
# the real `anthropic` package or a real API key.


class FakeGatedProvider(ModelProvider):
    capability = "model.anthropic"

    def __init__(self, response_text: str = "real answer"):
        self.response_text = response_text
        self.called = False

    def complete(self, prompt: str) -> ModelResponse:
        self.called = True
        return ModelResponse(text=self.response_text, confidence=None)


class FakeFailingProvider(ModelProvider):
    capability = "model.anthropic"

    def complete(self, prompt: str) -> ModelResponse:
        raise ConnectionError("simulated network failure")


def make_interpreter(model_name: str, provider: ModelProvider) -> tuple[Interpreter, list[str]]:
    out: list[str] = []
    interp = Interpreter(stdout=out.append)
    interp.globals.define(model_name, SenseModel(name=model_name, provider=provider))
    return interp, out


# -- inference("anthropic", ...): configuration -----------------------------------
# anthropic_model(...) was replaced by the provider-agnostic inference(...) --
# one function for every real vendor.


def test_inference_anthropic_without_api_key_raises_immediately():
    with pytest.raises(SenseRuntimeError, match="ANTHROPIC_API_KEY"):
        run('inference("anthropic")\n')


def test_inference_anthropic_respects_custom_env_var_name():
    with pytest.raises(SenseRuntimeError, match="MY_CUSTOM_KEY"):
        run('inference("anthropic", "claude-sonnet-5", api_key_env: "MY_CUSTOM_KEY")\n')


def test_inference_provider_must_be_a_string():
    with pytest.raises(SenseRuntimeError):
        run("inference(1)\n")


def test_inference_unknown_provider_raises():
    with pytest.raises(SenseRuntimeError, match="unknown provider"):
        run('inference("cohere")\n')


def test_inference_wrong_arity_raises():
    with pytest.raises(SenseRuntimeError):
        run('inference("anthropic", "claude-sonnet-5", "extra-positional")\n')


def test_inference_unknown_labeled_option_raises():
    with pytest.raises(SenseRuntimeError, match="unknown option"):
        run('inference("anthropic", "claude-sonnet-5", frobnicate: 1)\n')


def test_inference_anthropic_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-test")
    out = run('model m = inference("anthropic")\nprint(type_of(m))\n')
    assert out == ["Model"]


def test_model_keyword_names_the_result(monkeypatch):
    """`model NAME = inference(...)` -- the declaration itself names the
    model, no separate name argument needed."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-test")
    out = run('model claude = inference("anthropic")\nprint(claude.name)\n')
    assert out == ["claude"]


def test_model_keyword_rejects_a_non_model_value():
    with pytest.raises(SenseTypeError):
        run("model x = 5\n")


# -- capability gating: proven with a fake provider, no real package needed -------


def test_real_model_call_is_denied_by_policy_before_complete_runs():
    provider = FakeGatedProvider()
    interp, out = make_interpreter("claude", provider)
    with pytest.raises(SensePolicyError):
        interp.run_source(
            dedent(
                """
                policy:
                    deny model.anthropic
                set delegation = claude
                ask("hi")
                """
            )
        )
    assert provider.called is False


def test_real_model_call_succeeds_when_policy_allows():
    provider = FakeGatedProvider(response_text="real answer")
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            policy:
                allow model.anthropic
            set delegation = claude
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
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            set delegation = claude
            print(ask("hi").value)
            """
        )
    )
    assert out == ["real answer"]


def test_inference_mock_is_never_gated_even_under_a_deny_all_policy():
    # MockModelProvider has no `capability` attribute -- the gate is
    # opt-in per-provider, so this proves existing mock-model behavior is
    # completely unaffected by this feature.
    out = run(
        """
        model m = inference("mock", "demo")
        policy:
            deny model.anthropic
        set delegation = m
        print(ask("hi").value)
        """
    )
    assert out == ["(mock reasoning about: hi)"]


def test_explicit_model_dot_ask_is_also_gated():
    provider = FakeGatedProvider()
    interp, out = make_interpreter("claude", provider)
    with pytest.raises(SensePolicyError):
        interp.run_source(
            dedent(
                """
                policy:
                    deny model.anthropic
                claude.ask("hi")
                """
            )
        )
    assert provider.called is False


# -- errors never reach the user as a raw Python traceback -------------------------


def test_provider_failure_is_wrapped_not_a_raw_traceback():
    provider = FakeFailingProvider()
    interp, out = make_interpreter("claude", provider)
    with pytest.raises(SenseRuntimeError, match="ConnectionError"):
        interp.run_source(
            dedent(
                """
                policy:
                    allow model.anthropic
                set delegation = claude
                ask("hi")
                """
            )
        )


# -- audit log: only real (capability-bearing) calls are recorded -----------------


def test_real_model_calls_are_audited():
    provider = FakeGatedProvider()
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            policy:
                allow model.anthropic
            set delegation = claude
            ask("hi")
            for line in audit_log():
                print(line)
            """
        )
    )
    assert any("claude" in line and "called" in line for line in out)


def test_inference_mock_calls_are_not_audited():
    out = run(
        """
        model m = inference("mock", "demo")
        set delegation = m
        ask("hi")
        print(len(audit_log()))
        """
    )
    assert out == ["0"]


# -- exercising AnthropicProvider itself, only if the real package is installed.
# `importorskip` is called *inside* the test, not at module level -- at module
# level it would abort collection of this whole file when the package is
# missing, silently losing every offline test above it, not just this one.


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


def test_anthropic_provider_complete_parses_the_real_sdk_response_shape(monkeypatch):
    anthropic = pytest.importorskip("anthropic", reason="anthropic package not installed")
    from sense_lang.providers import AnthropicProvider

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeAnthropicResponse("hello from claude")

    class FakeAnthropicClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropicClient)

    provider = AnthropicProvider(model_id="claude-sonnet-5", api_key="sk-fake")
    response = provider.complete("what is 2+2?")

    assert response.text == "hello from claude"
    assert response.confidence is None
    assert captured["api_key"] == "sk-fake"
    assert captured["model"] == "claude-sonnet-5"
    assert captured["messages"] == [{"role": "user", "content": "what is 2+2?"}]
