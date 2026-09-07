"""`skill(name, description, tools)` -- a named, reusable bundle of `tool`s,
with no dedicated keyword: the `skill(...)` builtin already returns a
`Skill`, so the builtin's return type carries the meaning wherever it's
checked, here in `ask_with_tools`, without new grammar.

The description isn't just documentation: it's prepended to the prompt as
an "Available skills:" preamble when the skill is used, so it has real,
not cosmetic, effect -- tested below.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SenseRuntimeError
from sense_lang.interpreter import Interpreter
from sense_lang.providers import ModelProvider, ModelTurn, ToolCallRequest
from sense_lang.values import SenseModel


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


class FakeToolCallingProvider(ModelProvider):
    """Requests `tool_name` once, then answers using whatever came back."""

    capability = "model.anthropic"

    def __init__(self, tool_name: str, arguments: dict):
        self.tool_name = tool_name
        self.arguments = arguments
        self.turn = 0
        self.captured_messages = None

    def complete(self, prompt: str):
        raise NotImplementedError

    def complete_with_tools(self, messages, tools) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            self.captured_messages = messages
            return ModelTurn(
                text=None,
                tool_calls=[ToolCallRequest(id="c1", name=self.tool_name, arguments=self.arguments)],
                raw_assistant_message={"role": "assistant", "content": "requesting"},
            )
        last = messages[-1]["content"][0]["content"]
        return ModelTurn(text=f"final: {last}")

    def format_tool_results(self, results):
        # This fake declares capability = "model.anthropic" and reads
        # that exact bundled shape back out of messages[-1] above.
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


# -- construction ---------------------------------------------------------------


def test_skill_returns_a_skill_value_with_readable_members():
    out = run(
        """
        tool search(query: String) returns String "search the web":
            return "result"
        s = skill("web_research", "search the web to answer questions", [search])
        print(type_of(s))
        print(s.name)
        print(s.description)
        print(len(s.tools))
        """
    )
    assert out == ["Skill", "web_research", "search the web to answer questions", "1"]


def test_skill_can_bundle_multiple_tools():
    out = run(
        """
        tool search(query: String) returns String "search the web":
            return "result"
        tool fetch_page(url: String) returns String "fetch a page":
            return "page"
        s = skill("web_research", "search and fetch", [search, fetch_page])
        print(len(s.tools))
        """
    )
    assert out == ["2"]


# -- validation, same checks ask_with_tools() applies, caught earlier ---------


def test_skill_with_no_tools_is_rejected():
    with pytest.raises(SenseRuntimeError, match="at least one tool"):
        run('skill("s", "desc", [])\n')


def test_skill_tools_argument_must_be_an_array():
    with pytest.raises(SenseRuntimeError, match="Array"):
        run('skill("s", "desc", "not an array")\n')


def test_skill_rejects_a_bare_fn():
    with pytest.raises(SenseRuntimeError, match="only tools declared with"):
        run(
            """
            def add(a: Int, b: Int) -> Int:
                return a + b
            skill("s", "desc", [add])
            """
        )


def test_skill_rejects_an_async_tool():
    with pytest.raises(SenseRuntimeError, match="async"):
        run(
            """
            async tool bad(x: Int) returns Int "does a thing":
                return x
            skill("s", "desc", [bad])
            """
        )


def test_skill_rejects_a_tool_with_no_description():
    with pytest.raises(SenseRuntimeError, match="no description"):
        run(
            """
            tool bad(x: Int) returns Int:
                return x
            skill("s", "desc", [bad])
            """
        )


def test_skill_name_and_description_must_be_strings():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            tool ok(x: Int) returns Int "does a thing":
                return x
            skill(1, "desc", [ok])
            """
        )
    with pytest.raises(SenseRuntimeError):
        run(
            """
            tool ok(x: Int) returns Int "does a thing":
                return x
            skill("s", 1, [ok])
            """
        )


def test_skill_wrong_arity_raises():
    with pytest.raises(SenseRuntimeError):
        run('skill("s", "desc")\n')


# -- integration with ask_with_tools: flattening + description injection -----


def test_ask_with_tools_flattens_a_skill_into_its_tools():
    provider = FakeToolCallingProvider(tool_name="search", arguments={"query": "weather"})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool search(query: String) returns String requires web.search "search the web":
                return "sunny"

            policy:
                allow web.search

            web_research = skill("web_research", "search the web to answer questions", [search])
            set delegation = claude
            result = ask_with_tools("what's the weather?", [web_research])
            print(result.value)
            """
        )
    )
    assert out == ["final: sunny"]


def test_skill_description_is_injected_into_the_prompt():
    provider = FakeToolCallingProvider(tool_name="search", arguments={"query": "x"})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool search(query: String) returns String "search the web":
                return "result"
            web_research = skill("web_research", "search the web to answer questions", [search])
            set delegation = claude
            ask_with_tools("original question", [web_research])
            """
        )
    )
    first_prompt = provider.captured_messages[0]["content"]
    assert "Available skills:" in first_prompt
    assert "web_research: search the web to answer questions" in first_prompt
    assert first_prompt.endswith("original question")


def test_multiple_skills_each_contribute_a_description_line():
    class MultiSkillProvider(ModelProvider):
        capability = "model.anthropic"

        def complete(self, prompt):
            raise NotImplementedError

        def complete_with_tools(self, messages, tools):
            self.first_prompt = messages[0]["content"]
            return ModelTurn(text="done")

    provider = MultiSkillProvider()
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool a(x: Int) returns Int "tool a":
                return x
            tool b(x: Int) returns Int "tool b":
                return x
            skill_a = skill("skill_a", "does a things", [a])
            skill_b = skill("skill_b", "does b things", [b])
            set delegation = claude
            ask_with_tools("q", [skill_a, skill_b])
            """
        )
    )
    assert "skill_a: does a things" in provider.first_prompt
    assert "skill_b: does b things" in provider.first_prompt


def test_mixing_a_skill_with_a_bare_tool_works():
    class RecordingProvider(ModelProvider):
        capability = "model.anthropic"

        def complete(self, prompt):
            raise NotImplementedError

        def complete_with_tools(self, messages, tools):
            self.tool_names = sorted(t.name for t in tools)
            return ModelTurn(text="done")

    provider = RecordingProvider()
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool a(x: Int) returns Int "tool a":
                return x
            tool standalone(x: Int) returns Int "a standalone tool":
                return x
            bundle = skill("bundle", "a bundle", [a])
            set delegation = claude
            ask_with_tools("q", [bundle, standalone])
            """
        )
    )
    assert provider.tool_names == ["a", "standalone"]


def test_duplicate_tool_name_across_skill_and_bare_tool_is_rejected():
    interp, out = make_interpreter("claude", FakeToolCallingProvider("x", {}))
    with pytest.raises(SenseRuntimeError, match="duplicate tool name"):
        interp.run_source(
            dedent(
                """
                tool search(query: String) returns String "search the web":
                    return "result"
                bundle = skill("bundle", "a bundle", [search])
                set delegation = claude
                ask_with_tools("q", [bundle, search])
                """
            )
        )
