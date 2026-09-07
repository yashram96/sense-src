"""`ask_with_tools(prompt, tools)` -- a real model choosing which `tool`
to invoke, the first time a model's own decision (not human-written Sense
control flow) reaches `Interpreter._call_tool`. Every test here runs fully
offline via a small fake `ModelProvider` that decides deterministically
which tool to "request," proving the orchestration loop and (most
importantly) that a policy-denied tool the model asks for is never
executed -- the actual thesis this feature exists to test.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseRuntimeError
from sense_lang.interpreter import Interpreter
from sense_lang.providers import ModelProvider, ModelTurn, ToolCallRequest
from sense_lang.values import SenseModel


class FakeToolCallingProvider(ModelProvider):
    """Deterministic, offline stand-in for a model that supports tool
    calling: on the first turn it requests the named tool with the given
    arguments; on every later turn it returns a final answer built from
    whatever the most recent tool result was."""

    capability = "model.anthropic"

    def __init__(self, tool_name: str, arguments: dict, num_calls: int = 1):
        self.tool_name = tool_name
        self.arguments = arguments
        self.num_calls = num_calls
        self.turns = 0

    def complete(self, prompt: str):
        raise NotImplementedError

    def complete_with_tools(self, messages, tools) -> ModelTurn:
        self.turns += 1
        if self.turns <= self.num_calls:
            return ModelTurn(
                text=None,
                tool_calls=[ToolCallRequest(id=f"call{self.turns}", name=self.tool_name, arguments=self.arguments)],
                raw_assistant_message={"role": "assistant", "content": f"requesting {self.tool_name}"},
            )
        last_tool_result = messages[-1]["content"][0]["content"]
        return ModelTurn(text=f"final answer based on: {last_tool_result}")

    def format_tool_results(self, results):
        # Anthropic's own shape (one bundled user message) -- this fake
        # declares capability = "model.anthropic", and its own
        # complete_with_tools above reads that exact shape back out of
        # `messages[-1]`, so it needs to build the same thing.
        return [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": cid, "content": text} for cid, text in results],
            }
        ]


class AlwaysLoopsProvider(ModelProvider):
    capability = "model.anthropic"

    def complete(self, prompt: str):
        raise NotImplementedError

    def complete_with_tools(self, messages, tools) -> ModelTurn:
        return ModelTurn(
            text=None,
            tool_calls=[ToolCallRequest(id="c", name=tools[0].name, arguments={})],
            raw_assistant_message={"role": "assistant", "content": "loop"},
        )

    def format_tool_results(self, results):
        return [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": cid, "content": text} for cid, text in results],
            }
        ]


def make_interpreter(model_name: str, provider: ModelProvider) -> tuple[Interpreter, list[str]]:
    out: list[str] = []
    interp = Interpreter(stdout=out.append)
    interp.globals.define(model_name, SenseModel(name=model_name, provider=provider))
    return interp, out


# -- happy path ---------------------------------------------------------------------


def test_model_calls_a_tool_and_returns_a_final_answer():
    provider = FakeToolCallingProvider(tool_name="search", arguments={"query": "weather in paris"})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool search(query: String) returns String requires web.search "search the web":
                return "sunny, 22C"

            policy:
                allow web.search

            set delegation = claude
            result = ask_with_tools("what is the weather in paris", [search])
            print(result.value)
            print(result.confidence)
            """
        )
    )
    assert out == ["final answer based on: sunny, 22C", "nil"]


def test_tool_calls_within_the_loop_are_audited():
    provider = FakeToolCallingProvider(tool_name="search", arguments={"query": "x"})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool search(query: String) returns String requires web.search "search the web":
                return "result"

            policy:
                allow web.search

            set delegation = claude
            ask_with_tools("q", [search])
            for line in audit_log():
                print(line)
            """
        )
    )
    assert any("tool 'search' -> called" in line for line in out)
    assert any("model 'claude' -> called" in line for line in out)


# -- the actual safety proof: a policy-denied tool is never executed ---------------


def test_policy_denied_tool_request_is_refused_not_executed():
    provider = FakeToolCallingProvider(tool_name="charge", arguments={"amount": 100})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            memory side_effects

            tool charge(amount: Int) returns String requires payment.execute "charge a payment":
                side_effects.remember("charged", amount)
                return "charged " + str(amount)

            policy:
                deny payment.execute

            set delegation = claude
            result = ask_with_tools("charge the customer", [charge])
            print(result.value)
            print(side_effects.recall("charged"))
            for line in audit_log():
                print(line)
            """
        )
    )
    # The tool's body -- specifically its side effect -- never ran.
    assert out[1] == "nil"
    # ask_with_tools() itself did not crash; the error was fed back to
    # the (fake) model, which responded based on it.
    assert "error" in out[0]
    assert "denied by policy" in out[0]
    # The denial is on the record, same as a hand-written denied call.
    assert any("tool 'charge' -> denied" in line for line in out)


def test_policy_allowed_tool_from_the_model_actually_runs():
    provider = FakeToolCallingProvider(tool_name="charge", arguments={"amount": 50})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            memory side_effects

            tool charge(amount: Int) returns String requires payment.execute "charge a payment":
                side_effects.remember("charged", amount)
                return "charged " + str(amount)

            policy:
                allow payment.execute

            set delegation = claude
            ask_with_tools("charge the customer", [charge])
            print(side_effects.recall("charged"))
            """
        )
    )
    assert out == ["50"]


def test_model_requesting_a_tool_not_in_the_list_is_fed_back_as_an_error():
    provider = FakeToolCallingProvider(tool_name="nonexistent", arguments={})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool search(query: String) returns String "search the web":
                return "result"

            set delegation = claude
            result = ask_with_tools("q", [search])
            print(result.value)
            """
        )
    )
    assert "no such tool" in out[0]


def test_bad_tool_arguments_from_the_model_are_fed_back_not_raised():
    # The model provides a String where the tool declares an Int -- this
    # should surface as a fed-back error, not a crash, giving the model a
    # chance to retry with corrected arguments (which this fake doesn't
    # do, but the call must not propagate as SenseTypeError to Sense code).
    provider = FakeToolCallingProvider(tool_name="add", arguments={"a": "not a number", "b": 2})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool add(a: Int, b: Int) returns Int "adds two numbers":
                return a + b

            set delegation = claude
            result = ask_with_tools("add", [add])
            print(result.value)
            """
        )
    )
    assert "error" in out[0]


# -- validation of the tools list ---------------------------------------------------


def test_tool_with_no_description_is_rejected():
    interp, out = make_interpreter("claude", FakeToolCallingProvider("x", {}))
    with pytest.raises(SenseRuntimeError, match="no description"):
        interp.run_source(
            dedent(
                """
                tool bad(x: Int) returns Int:
                    return x
                set delegation = claude
                ask_with_tools("x", [bad])
                """
            )
        )


def test_async_tool_is_rejected():
    interp, out = make_interpreter("claude", FakeToolCallingProvider("x", {}))
    with pytest.raises(SenseRuntimeError, match="async"):
        interp.run_source(
            dedent(
                """
                async tool bad(x: Int) returns Int "does a thing":
                    return x
                set delegation = claude
                ask_with_tools("x", [bad])
                """
            )
        )


def test_non_tool_value_in_the_list_is_rejected():
    interp, out = make_interpreter("claude", FakeToolCallingProvider("x", {}))
    with pytest.raises(SenseRuntimeError, match="only tools declared with"):
        interp.run_source(
            dedent(
                """
                def add(a: Int, b: Int) -> Int:
                    return a + b
                set delegation = claude
                ask_with_tools("x", [add])
                """
            )
        )


def test_tools_argument_must_be_an_array():
    interp, out = make_interpreter("claude", FakeToolCallingProvider("x", {}))
    with pytest.raises(SenseRuntimeError, match="Array"):
        interp.run_source(
            dedent(
                """
                set delegation = claude
                ask_with_tools("x", "not an array")
                """
            )
        )


# -- provider support / policy / iteration cap --------------------------------------


def test_provider_without_tool_calling_support_raises_clearly():
    out = []
    interp = Interpreter(stdout=out.append)
    with pytest.raises(SenseRuntimeError, match="does not support"):
        interp.run_source(
            dedent(
                """
                tool search(query: String) returns String "search the web":
                    return "result"
                model m = inference("mock", "demo")
                set delegation = m
                ask_with_tools("q", [search])
                """
            )
        )


def test_ask_with_tools_is_capability_gated_before_any_call():
    from sense_lang.errors import SensePolicyError

    provider = FakeToolCallingProvider("x", {})
    interp, out = make_interpreter("claude", provider)
    with pytest.raises(SensePolicyError):
        interp.run_source(
            dedent(
                """
                tool search(query: String) returns String "search the web":
                    return "result"
                policy:
                    deny model.anthropic
                set delegation = claude
                ask_with_tools("q", [search])
                """
            )
        )
    assert provider.turns == 0


def test_iteration_cap_prevents_an_infinite_loop():
    interp, out = make_interpreter("claude", AlwaysLoopsProvider())
    with pytest.raises(SenseRuntimeError, match="tool-call iterations"):
        interp.run_source(
            dedent(
                """
                tool search(query: String) returns String "search the web":
                    return "result"
                set delegation = claude
                ask_with_tools("q", [search])
                """
            )
        )


def test_type_of_result_is_answer():
    provider = FakeToolCallingProvider(tool_name="search", arguments={"query": "x"})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool search(query: String) returns String "search the web":
                return "result"
            set delegation = claude
            result = ask_with_tools("q", [search])
            print(type_of(result))
            """
        )
    )
    assert out == ["Answer"]


# -- exercising AnthropicProvider.complete_with_tools, only if installed ----------


def test_anthropic_provider_complete_with_tools_parses_tool_use_response(monkeypatch):
    anthropic = pytest.importorskip("anthropic", reason="anthropic package not installed")
    from sense_lang.providers import AnthropicProvider, ToolSpec

    class _FakeToolUseBlock:
        def __init__(self, id_, name, input_):
            self.type = "tool_use"
            self.id = id_
            self.name = name
            self.input = input_

    class _FakeResponse:
        def __init__(self, content):
            self.content = content

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse([_FakeToolUseBlock("call_1", "search", {"query": "weather"})])

    class FakeAnthropicClient:
        def __init__(self, api_key):
            self.messages = FakeMessages()

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropicClient)

    provider = AnthropicProvider(model_id="claude-sonnet-5", api_key="sk-fake")
    spec = ToolSpec(name="search", description="search the web", parameters={"type": "object", "properties": {}})
    turn = provider.complete_with_tools([{"role": "user", "content": "hi"}], [spec])

    assert turn.text is None
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "search"
    assert turn.tool_calls[0].arguments == {"query": "weather"}
    assert captured["tools"] == [
        {"name": "search", "description": "search the web", "input_schema": {"type": "object", "properties": {}}}
    ]
