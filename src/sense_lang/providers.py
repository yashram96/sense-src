"""Model provider abstraction.

`ask(...)` and `<model>.ask(...)` must not hard-code a vendor — the
runtime decides how a reasoning capability is implemented. This module
defines the seam (`ModelProvider`) and ships three concrete providers:
`MockModelProvider`, so Sense programs stay runnable offline with no API
key, and two real vendor
backends behind the identical interface, `AnthropicProvider` and
`OpenAIProvider`. Both are reached through one Sense-level builtin,
`inference(provider, model_id?, ...)` (`Interpreter`'s `_inference`) --
first shipped as two separate provider-specific builtins
(`anthropic_model(...)`, `openai_model(...)`), consolidated once a second
real provider made "one function per vendor" the wrong call. Adding that
second provider did surface one thing that genuinely needed
generalizing, narrower than the whole dispatcher this module used to say
would wait for it: only `ModelTurn.raw_assistant_message` was already
provider-opaque, but the *tool-result* message `Interpreter._ask_with_tools`
built to append after running a model's requested tools was hardcoded to
Anthropic's own wire shape (`{"type": "tool_result", ...}` inside one
user message). OpenAI's Chat Completions API wants something structurally
different -- one separate `{"role": "tool", "tool_call_id": ...}` message
per call, not one message bundling all of them -- so that step is now
`ModelProvider.format_tool_results(...)`, a provider's own concern, the
same way `raw_assistant_message`'s shape already was. Nothing else needed
to move: `complete`/`complete_with_tools`'s signatures were already
vendor-agnostic.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    text: str
    confidence: float | None = None


@dataclass
class ToolSpec:
    """One `tool` offered to a model for `ask_with_tools(...)` --
    built by `Interpreter._build_tool_spec` from a `SenseToolDef`'s
    params/type annotations. `parameters` is a JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCallRequest:
    """One tool invocation a model asked for, in one turn. `id` is the
    provider's own call id, threaded back unchanged in the tool-result
    message so the model can match results to requests."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    """The result of one `complete_with_tools` step -- one call in, one
    turn out, regardless of vendor. Exactly one of `text`/`tool_calls` is
    meaningful: `text` set means the model is done; otherwise `tool_calls`
    lists what it wants to run next. `raw_assistant_message` is opaque
    provider-specific state (that provider's own message shape -- an
    Anthropic `content`-blocks dict, an OpenAI `tool_calls`-bearing dict)
    appended to the conversation as-is by `Interpreter._ask_with_tools`
    -- the interpreter never inspects it, only the provider that produced
    it does, on the next turn."""

    text: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    raw_assistant_message: Any = None


class ModelProvider(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> ModelResponse:
        raise NotImplementedError

    def complete_with_tools(self, messages: list[Any], tools: list[ToolSpec]) -> ModelTurn:
        """One step of an LLM-driven tool-calling loop -- see
        `Interpreter._ask_with_tools`, which owns the loop and the
        growing `messages` list; providers are stateless across calls.
        Not implemented by default: simulating a real model's tool-
        selection judgement with a canned template (`MockModelProvider`)
        would be dishonest, not a useful mock."""
        raise NotImplementedError(f"{type(self).__name__} does not support tool-calling")

    def format_tool_results(self, results: list[tuple[str, str]]) -> list[Any]:
        """Given the `(tool_call_id, result_text)` pairs from one turn's
        worth of executed tool calls, build the message(s) to append to
        the conversation before the next `complete_with_tools()` call --
        each provider's own wire shape for "here's what those tools
        returned." Anthropic wants one `user` message bundling every
        result as `tool_result` content blocks; OpenAI wants one separate
        `{"role": "tool", ...}` message per call -- hence a *list* of
        messages, not one. Not implemented by default, same reasoning
        `complete_with_tools` itself isn't."""
        raise NotImplementedError(f"{type(self).__name__} does not support tool-calling")


class MockModelProvider(ModelProvider):
    """Deterministic, offline stand-in for a real reasoning model.

    Useful for testing Sense programs (and this interpreter) without network
    access or API keys. `response_template` may use `{prompt}`.
    """

    def __init__(self, response_template: str = "(mock reasoning about: {prompt})", confidence: float = 0.5):
        self.response_template = response_template
        self.confidence = confidence

    def complete(self, prompt: str) -> ModelResponse:
        return ModelResponse(text=self.response_template.format(prompt=prompt), confidence=self.confidence)


class AnthropicProvider(ModelProvider):
    """A real provider, behind the exact same `ModelProvider` seam --
    "the runtime decides how reasoning is implemented" holds even here: a
    Sense program calling `ask()` cannot tell whether it's talking to
    this or `MockModelProvider`.

    `capability`: a real call costs money and sends prompt data to a third
    party over the network -- exactly the shape of thing `policy`/
    `requires` already exist for (`tool`/`action` use the identical
    mechanism). `Interpreter._ask_model` checks this attribute
    (absent on `MockModelProvider`, so mock calls are never gated) against
    the caller's policy scope before every `complete()` call.

    Uses the official `anthropic` SDK (an optional dependency -- imported
    lazily here, not at module load, so importing `sense_lang` itself
    never requires it) rather than hand-rolled HTTP: there's no stdlib
    equivalent for a vendor's API contract, and reinventing retries/
    timeouts/auth-error handling by hand would just be worse than what the
    SDK already does.
    """

    capability = "model.anthropic"

    def __init__(self, model_id: str, api_key: str, temperature: float | None = None, max_tokens: int | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "the 'anthropic' package is required for inference(\"anthropic\", ...) -- "
                "install it with 'pip install anthropic' (or 'pip install sense-lang[anthropic]')"
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        # Anthropic's API requires max_tokens on every call, unlike
        # OpenAI's (optional there) -- 1024 is the same default this
        # provider always used before temperature/max_tokens became
        # configurable via inference(...)'s labeled arguments.
        self.max_tokens = max_tokens if max_tokens is not None else 1024

    def _sampling_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        return kwargs

    def complete(self, prompt: str) -> ModelResponse:
        response = self._client.messages.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            **self._sampling_kwargs(),
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        # A real model doesn't report a calibrated confidence the way
        # MockModelProvider fabricates one -- `None` is the honest answer,
        # and stringify() already renders a nil-confidence Answer as
        # "answer<?>(...)" (values.py), not a fake number.
        return ModelResponse(text=text, confidence=None)

    def complete_with_tools(self, messages: list, tools: list[ToolSpec]) -> ModelTurn:
        response = self._client.messages.create(
            model=self.model_id,
            messages=messages,
            tools=[
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ],
            **self._sampling_kwargs(),
        )
        tool_calls = [
            ToolCallRequest(id=block.id, name=block.name, arguments=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        # Anthropic's `stop_reason` names the turn, but inspecting the
        # content blocks directly (same as complete()'s text-joining
        # above) is the more robust signal: any tool_use block present
        # means the model wants to act before answering.
        assistant_message = {"role": "assistant", "content": response.content}
        if tool_calls:
            return ModelTurn(text=None, tool_calls=tool_calls, raw_assistant_message=assistant_message)
        text = "".join(block.text for block in response.content if block.type == "text")
        return ModelTurn(text=text, raw_assistant_message=assistant_message)

    def format_tool_results(self, results: list[tuple[str, str]]) -> list[Any]:
        return [
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": call_id, "content": text} for call_id, text in results],
            }
        ]


class OpenAIProvider(ModelProvider):
    """A second real provider behind the identical `ModelProvider` seam --
    `inference("openai", ...)` (the same function `inference("anthropic",
    ...)` uses, just a different `provider` argument) is the only new
    surface a Sense program sees; `ask()`/`ask_with_tools()`
    themselves don't change at all, proving the seam actually generalizes
    rather than being written Anthropic-specifically and hoping.

    `capability`: same reasoning as `AnthropicProvider.capability` -- a
    real call costs money and sends prompt data to a third party, gated
    like any other capability, checked by the exact same
    `Interpreter._ask_model`/`_ask_with_tools` code (which
    reads this attribute generically via `getattr`, never a vendor name).

    Uses the official `openai` SDK (an optional dependency, imported
    lazily so importing `sense_lang` itself never requires it) for the
    same reason `AnthropicProvider` uses `anthropic`: no stdlib equivalent
    for a vendor's API contract, and hand-rolling retries/auth-error
    handling would just be worse than what the SDK already does.
    """

    capability = "model.openai"

    def __init__(self, model_id: str, api_key: str, temperature: float | None = None, max_tokens: int | None = None):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "the 'openai' package is required for inference(\"openai\", ...) -- "
                "install it with 'pip install openai' (or 'pip install sense-lang[openai]')"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens  # unlike Anthropic, optional here -- omitted entirely if not given

    def _sampling_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            # OpenAI's wire parameter, not Sense's: `max_tokens` is rejected
            # outright by newer models ("Unsupported parameter: 'max_tokens'
            # ... Use 'max_completion_tokens' instead"), so this provider
            # always sends the newer name regardless of which model_id is
            # selected -- `inference(..., max_tokens: ...)` stays the one
            # Sense-level label across every provider; only the translation
            # to OpenAI's own wire format lives here.
            kwargs["max_completion_tokens"] = self.max_tokens
        return kwargs

    def complete(self, prompt: str) -> ModelResponse:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            **self._sampling_kwargs(),
        )
        text = response.choices[0].message.content or ""
        # Same honesty as AnthropicProvider: no calibrated confidence to
        # report, so `None` rather than a fabricated number.
        return ModelResponse(text=text, confidence=None)

    def complete_with_tools(self, messages: list, tools: list[ToolSpec]) -> ModelTurn:
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=[
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ],
            **self._sampling_kwargs(),
        )
        message = response.choices[0].message
        # The assistant message must be echoed back to OpenAI in exactly
        # this shape on the next turn (its own `tool_calls` format, not
        # the SDK's response objects) -- this is what
        # ModelTurn.raw_assistant_message exists for.
        assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
            tool_calls = [
                ToolCallRequest(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
                for tc in message.tool_calls
            ]
            return ModelTurn(text=None, tool_calls=tool_calls, raw_assistant_message=assistant_message)
        return ModelTurn(text=message.content or "", raw_assistant_message=assistant_message)

    def format_tool_results(self, results: list[tuple[str, str]]) -> list[Any]:
        # Unlike Anthropic's one bundled message, OpenAI wants one
        # separate `tool`-role message per call, each naming which
        # `tool_call_id` it answers.
        return [{"role": "tool", "tool_call_id": call_id, "content": text} for call_id, text in results]
