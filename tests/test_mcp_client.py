"""MCP client support (`connect_mcp`) -- the fifth and last piece of the
"real LLM integration" ask, and the one genuinely new piece of runtime
infrastructure in it: an MCP tool has no Sense body, so invoking one is a
JSON-RPC call to an external process, bridged from Sense's synchronous
interpreter via a background thread running its own asyncio event loop
(`mcp_client.McpBridge`).

Every test here uses a **real** MCP server (`tests/mcp_test_server.py`,
run as a real subprocess) rather than a mock -- the whole point of this
feature is a real external process, and the riskiest part (the async/sync
bridge) is exactly what a mock would fail to exercise honestly. Skipped
entirely if the `mcp` package isn't installed (checked inside each test
that needs it, not at module level -- a module-level `importorskip` would
abort collection of every test in this file, not just the ones needing
the package, a bug already found and fixed in test_anthropic_provider.py
earlier this session).
"""

import sys
from pathlib import Path
from textwrap import dedent

import pytest

from sense_lang.errors import SensePolicyError, SenseRuntimeError
from sense_lang.interpreter import Interpreter
from sense_lang.providers import ModelProvider, ModelTurn, ToolCallRequest
from sense_lang.values import SenseModel

SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_test_server.py")


def run(source: str) -> list[str]:
    output: list[str] = []
    interp = Interpreter(stdout=output.append)
    interp.globals.define("python_exe", sys.executable)
    interp.globals.define("server_script", SERVER_SCRIPT)
    interp.run_source(dedent(source))
    return output


class FakeToolCallingProvider(ModelProvider):
    """Requests the first tool it's offered, once, with the given
    arguments, then answers using whatever the tool result was."""

    capability = "model.anthropic"

    def __init__(self, arguments: dict, tool_index: int = 0):
        self.arguments = arguments
        self.tool_index = tool_index
        self.turn = 0
        self.captured_tools = None

    def complete(self, prompt: str):
        raise NotImplementedError

    def complete_with_tools(self, messages, tools) -> ModelTurn:
        self.turn += 1
        if self.turn == 1:
            self.captured_tools = tools
            return ModelTurn(
                text=None,
                tool_calls=[
                    ToolCallRequest(id="c1", name=tools[self.tool_index].name, arguments=self.arguments)
                ],
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
    interp.globals.define("python_exe", sys.executable)
    interp.globals.define("server_script", SERVER_SCRIPT)
    return interp, out


# -- connect_mcp(): a real subprocess, a real handshake ----------------------------


def test_connect_mcp_discovers_real_tools():
    pytest.importorskip("mcp", reason="mcp package not installed")
    out = run(
        """
        server = connect_mcp("calc", python_exe, [server_script])
        print(type_of(server))
        print(server.name)
        print(len(server.tools))
        print(type_of(server.tools[0]))
        server.close()
        """
    )
    assert out == ["McpServer", "calc", "2", "Tool"]


def test_connect_mcp_wrong_arity_raises():
    pytest.importorskip("mcp", reason="mcp package not installed")
    with pytest.raises(SenseRuntimeError):
        run('connect_mcp("only-one-arg")\n')


def test_connect_mcp_argument_types_are_checked():
    pytest.importorskip("mcp", reason="mcp package not installed")
    with pytest.raises(SenseRuntimeError):
        run('connect_mcp(1, "cmd", [])\n')
    with pytest.raises(SenseRuntimeError):
        run('connect_mcp("name", 1, [])\n')
    with pytest.raises(SenseRuntimeError):
        run('connect_mcp("name", "cmd", "not an array")\n')


def test_connect_mcp_bad_command_fails_clearly_not_hanging():
    pytest.importorskip("mcp", reason="mcp package not installed")
    with pytest.raises(SenseRuntimeError, match="failed to connect"):
        run('connect_mcp("bad", "this-command-does-not-exist-xyz", [])\n')


def test_close_stops_the_bridge_thread():
    pytest.importorskip("mcp", reason="mcp package not installed")
    from sense_lang.mcp_client import McpBridge

    bridge = McpBridge(sys.executable, [SERVER_SCRIPT])
    bridge.start()
    assert bridge._thread.is_alive()
    bridge.close()
    assert not bridge._thread.is_alive()


# -- calling a real tool through ask_with_tools ---------------------------------


def test_ask_with_tools_calls_a_real_mcp_tool():
    pytest.importorskip("mcp", reason="mcp package not installed")
    provider = FakeToolCallingProvider(arguments={"a": 4, "b": 5})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            server = connect_mcp("calc", python_exe, [server_script], "mcp.calc")
            policy:
                allow mcp.calc
            set delegation = claude
            result = ask_with_tools("add 4 and 5", server.tools)
            print(result.value)
            server.close()
            """
        )
    )
    assert out == ["final: 9"]


def test_mcp_tool_name_is_qualified_by_server_name():
    pytest.importorskip("mcp", reason="mcp package not installed")
    provider = FakeToolCallingProvider(arguments={"a": 1, "b": 1})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            server = connect_mcp("calc", python_exe, [server_script])
            set delegation = claude
            ask_with_tools("add", server.tools)
            server.close()
            """
        )
    )
    assert provider.captured_tools[0].name == "calc.add"


def test_mcp_tool_calls_are_audited():
    pytest.importorskip("mcp", reason="mcp package not installed")
    provider = FakeToolCallingProvider(arguments={"a": 1, "b": 1})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            server = connect_mcp("calc", python_exe, [server_script])
            set delegation = claude
            ask_with_tools("add", server.tools)
            for line in audit_log():
                print(line)
            server.close()
            """
        )
    )
    assert any("calc.add" in line and "called" in line for line in out)


# -- the actual safety proof: capability denial blocks the real call --------------


def test_policy_denied_mcp_tool_is_never_invoked():
    pytest.importorskip("mcp", reason="mcp package not installed")
    provider = FakeToolCallingProvider(arguments={"a": 1, "b": 2})
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            server = connect_mcp("calc", python_exe, [server_script], "mcp.calc")
            policy:
                deny mcp.calc
            set delegation = claude
            result = ask_with_tools("add 1 and 2", server.tools)
            print(result.value)
            for line in audit_log():
                print(line)
            server.close()
            """
        )
    )
    # The model was told no, not given a real answer -- "9" (a correct sum
    # for other inputs) never would have appeared; here the denial message
    # is what got fed back.
    assert "denied by policy" in out[0]
    assert any("calc.add" in line and "denied" in line for line in out)


def test_mcp_tool_failure_is_fed_back_not_raised():
    # tools[1] is "always_fails" -- proves a real remote tool-level
    # failure surfaces as a fed-back error, not an exception that crashes
    # ask_with_tools() itself.
    pytest.importorskip("mcp", reason="mcp package not installed")
    provider = FakeToolCallingProvider(arguments={}, tool_index=1)
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            server = connect_mcp("calc", python_exe, [server_script])
            set delegation = claude
            result = ask_with_tools("please fail", server.tools)
            print(result.value)
            server.close()
            """
        )
    )
    assert "error" in out[0].lower()


# -- mixing an McpTool with a bare tool and a skill ---------------------------------


def test_mcp_tool_mixes_with_a_bare_sense_tool():
    pytest.importorskip("mcp", reason="mcp package not installed")

    class MultiToolProvider(ModelProvider):
        capability = "model.anthropic"

        def complete(self, prompt):
            raise NotImplementedError

        def complete_with_tools(self, messages, tools):
            self.tool_names = sorted(t.name for t in tools)
            return ModelTurn(text="done")

    provider = MultiToolProvider()
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            tool standalone(x: Int) returns Int "a standalone tool":
                return x
            server = connect_mcp("calc", python_exe, [server_script])
            set delegation = claude
            ask_with_tools("q", [standalone, server.tools[0]])
            server.close()
            """
        )
    )
    assert provider.tool_names == ["calc.add", "standalone"]


def test_mcp_tool_can_be_bundled_into_a_skill():
    pytest.importorskip("mcp", reason="mcp package not installed")

    class RecordingProvider(ModelProvider):
        capability = "model.anthropic"

        def complete(self, prompt):
            raise NotImplementedError

        def complete_with_tools(self, messages, tools):
            self.prompt = messages[0]["content"]
            return ModelTurn(text="done")

    provider = RecordingProvider()
    interp, out = make_interpreter("claude", provider)
    interp.run_source(
        dedent(
            """
            server = connect_mcp("calc", python_exe, [server_script])
            calc_skill = skill("calc", "perform arithmetic", server.tools)
            set delegation = claude
            ask_with_tools("q", [calc_skill])
            server.close()
            """
        )
    )
    assert "calc: perform arithmetic" in provider.prompt


def test_type_of_mcp_server_and_tool():
    pytest.importorskip("mcp", reason="mcp package not installed")
    out = run(
        """
        server = connect_mcp("calc", python_exe, [server_script])
        print(type_of(server))
        print(type_of(server.tools[0]))
        server.close()
        """
    )
    assert out == ["McpServer", "Tool"]
