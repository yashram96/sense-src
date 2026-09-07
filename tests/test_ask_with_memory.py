"""`ask_with_memory(prompt, memory)` -- retrieval-into-prompt, the
lightweight half of "memory wired into reasoning." Write-back (the model
storing something back) is deliberately NOT built here -- it already
composes for free out of `memory`/`persistent memory` + `ask_with_tools`
(a small `tool` wrapping `.remember(...)`, handed to the model) -- an
automatic write-back guessing what to store from free text isn't
attempted here.
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


class FakeGatedProvider(ModelProvider):
    """A real-shaped provider: declares `capability`, so `ask_with_memory`
    gates it exactly like `inference("anthropic", ...)` -- proven without
    needing the real package, same pattern as test_anthropic_provider.py."""

    capability = "model.anthropic"

    def __init__(self):
        self.last_prompt: str | None = None

    def complete(self, prompt: str) -> ModelResponse:
        self.last_prompt = prompt
        return ModelResponse(text=f"answer: {prompt}", confidence=None)

    def complete_with_tools(self, messages, tools):
        raise NotImplementedError


def make_interpreter(model_name: str, provider: ModelProvider) -> tuple[Interpreter, list[str]]:
    out: list[str] = []
    interp = Interpreter(stdout=out.append)
    interp.globals.define(model_name, SenseModel(name=model_name, provider=provider))
    return interp, out


# -- session memory -----------------------------------------------------------------


def test_populated_memory_is_injected_as_context():
    out = run(
        """
        memory notes
        notes.remember("favorite_color", "blue")
        notes.remember("age", 36)

        model m = inference("mock", "demo", response_template: "I recall: {prompt}")
        set delegation = m

        print(ask_with_memory("what should I know?", notes).value)
        """
    )
    assert out == [
        "I recall: Known information from memory:\n"
        "- favorite_color: blue\n"
        "- age: 36\n\n"
        "what should I know?"
    ]


def test_empty_memory_leaves_the_prompt_unchanged():
    out = run(
        """
        memory notes
        model m = inference("mock", "demo", response_template: "reply: {prompt}")
        set delegation = m
        print(ask_with_memory("plain question", notes).value)
        """
    )
    assert out == ["reply: plain question"]


def test_forgotten_keys_are_not_included_in_context():
    out = run(
        """
        memory notes
        notes.remember("a", 1)
        notes.remember("b", 2)
        notes.forget("a")
        model m = inference("mock", "demo", response_template: "{prompt}")
        set delegation = m
        print(ask_with_memory("q", notes).value)
        """
    )
    assert "a:" not in out[0]
    assert "b: 2" in out[0]


# -- persistent memory ----------------------------------------------------------------


def test_persistent_memory_is_also_supported(tmp_path):
    db = str(tmp_path / "notes.db").replace("\\", "/")
    out = run(
        f"""
        persistent memory notes: "{db}"
        notes.remember("x", 1)
        model m = inference("mock", "demo", response_template: "{{prompt}}")
        set delegation = m
        print(ask_with_memory("q", notes).value)
        """
    )
    assert "x: 1" in out[0]


# -- capability gating: real providers only, same mechanism as ask() -----------


def test_ask_with_memory_is_gated_for_a_real_provider():
    provider = FakeGatedProvider()
    interp, out = make_interpreter("claude", provider)
    with pytest.raises(SensePolicyError):
        interp.run_source(
            dedent(
                """
                memory notes
                policy:
                    deny model.anthropic
                set delegation = claude
                ask_with_memory("q", notes)
                """
            )
        )
    assert provider.last_prompt is None


def test_ask_with_memory_succeeds_when_policy_allows():
    provider = FakeGatedProvider()
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            memory notes
            notes.remember("x", 1)
            policy:
                allow model.anthropic
            set delegation = claude
            print(ask_with_memory("q", notes).value)
            """
        )
    )
    assert "x: 1" in out[0]
    assert provider.last_prompt is not None


def test_inference_mock_is_never_gated():
    out = run(
        """
        memory notes
        model m = inference("mock", "demo")
        policy:
            deny model.anthropic
        set delegation = m
        print(type_of(ask_with_memory("q", notes)))
        """
    )
    assert out == ["Answer"]


# -- errors -------------------------------------------------------------------------


def test_non_memory_second_argument_is_rejected():
    with pytest.raises(SenseRuntimeError, match="Memory"):
        run(
            """
            model m = inference("mock", "demo")
            set delegation = m
            ask_with_memory("q", "not a memory")
            """
        )


def test_non_string_prompt_is_rejected():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            memory notes
            model m = inference("mock", "demo")
            set delegation = m
            ask_with_memory(42, notes)
            """
        )


def test_no_model_configured_raises():
    with pytest.raises(SenseRuntimeError, match="no model configured"):
        run(
            """
            memory notes
            ask_with_memory("q", notes)
            """
        )
