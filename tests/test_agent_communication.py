"""Inter-agent communication: `send(agent, message)` / `receive(timeout?)`,
a thread-safe mailbox per agent (`SenseAgent._mailbox`, a `queue.Queue`) --
previously agents could only share state by racing on a common enclosing
scope, with no synchronized way to pass a value between two independent
(especially `async`) agents.
"""

from __future__ import annotations

from textwrap import dedent

import pytest

from sense_lang.errors import SenseRuntimeError
from sense_lang.interpreter import Interpreter


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


def test_message_sent_before_start_is_waiting_when_the_agent_receives():
    out = run(
        """
        agent worker:
            task = receive()
            print("worker got: " + task)
        send(worker, "build the report")
        start(worker)
        """
    )
    assert out == ["worker got: build the report"]


def test_message_sent_to_a_paused_agent_is_waiting_when_it_resumes_into_receive():
    # pause() (not receive() itself) is what lets the caller's thread get
    # control back to run send() -- a sync agent blocked directly inside
    # receive() with no pause point would deadlock start()'s caller, since
    # nothing could ever run the send() needed to unblock it.
    out = run(
        """
        agent worker:
            pause()
            task = receive()
            print("worker got: " + task)
        start(worker)
        send(worker, "build the report")
        resume(worker)
        """
    )
    assert out == ["worker got: build the report"]


def test_two_async_agents_exchange_messages_both_directions():
    out = run(
        """
        async agent worker:
            task = receive()
            print("worker got: " + task)
            send(boss, "done: " + task)

        async agent boss:
            send(worker, "build the report")
            reply = receive()
            print("boss got: " + reply)

        start(worker)
        start(boss)
        await(worker)
        await(boss)
        """
    )
    assert out == ["worker got: build the report", "boss got: done: build the report"]


def test_messages_are_delivered_fifo():
    out = run(
        """
        agent worker:
            print(receive())
            print(receive())
            print(receive())
        send(worker, "first")
        send(worker, "second")
        send(worker, "third")
        start(worker)
        """
    )
    assert out == ["first", "second", "third"]


def test_receive_timeout_returns_nil_when_no_message_arrives():
    out = run(
        """
        agent lonely:
            msg = receive(0.05)
            print("got: " + str(msg))
        start(lonely)
        """
    )
    assert out == ["got: nil"]


def test_receive_outside_any_agent_raises():
    with pytest.raises(SenseRuntimeError, match="receive\\(\\) can only be called inside a running agent's body"):
        run("receive()")


def test_send_to_a_non_agent_raises():
    with pytest.raises(SenseRuntimeError, match="send\\(\\) expects an Agent"):
        run('send(5, "x")')


def test_one_agents_body_can_message_a_sibling_agent():
    """A message sent from *inside* another agent's body, not just from
    top-level script code -- the mailbox is just a queue, indifferent to
    who's putting into it."""
    out = run(
        """
        agent worker:
            print(receive())

        agent coordinator:
            send(worker, "from the coordinator")

        start(coordinator)
        start(worker)
        """
    )
    assert out == ["from the coordinator"]
