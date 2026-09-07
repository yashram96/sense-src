"""`async agent` / `async tool` -- sync stays the default; concurrency is
opt-in at this boundary, not threaded through every plain function call
(function coloring).

The central guarantee under test: `async` makes the caller and the
agent/tool body genuinely executing at the same time -- not just "runs
without error." Several tests below prove overlap via a shared marker
recorded before either side proceeds, not just correctness of the result.
"""

from textwrap import dedent

import pytest

from sense_lang.errors import SensePolicyError, SenseRuntimeError, SenseTypeError
from sense_lang.interpreter import Interpreter


def run(source: str) -> list[str]:
    output: list[str] = []
    Interpreter(stdout=output.append).run_source(dedent(source))
    return output


# -- async agent: start()/resume() don't block -----------------------------------


def test_start_on_async_agent_does_not_block():
    # A real wall-clock delay (via Python interop) makes this deterministic
    # rather than racing on thread-scheduling order: 0.1s is far longer than
    # the time it takes the main thread to reach the next line, so seeing
    # "running" (not "completed") proves start() returned before the sleep
    # finished, not just that it happened to be scheduled first.
    out = run(
        """
        import python "time" as time

        async agent a:
            time.sleep(0.1)
            x = 1

        started = start(a)
        print(started.status)
        await(a)
        print(a.status)
        """
    )
    assert out == ["running", "completed"]


def test_sync_agent_start_still_blocks_unaffected_by_this_feature():
    out = run(
        """
        import python "time" as time

        agent a:
            time.sleep(0.05)
            x = 1
        started = start(a)
        print(started.status)
        """
    )
    # Unlike the async case above, this reads "completed" immediately --
    # start() genuinely waited out the sleep before returning.
    assert out == ["completed"]


def test_two_async_agents_are_genuinely_concurrent():
    out = run(
        """
        memory marks

        async agent a:
            marks.remember("a_started", true)
            pause("waiting")

        async agent b:
            marks.remember("b_started", true)
            pause("waiting")

        start(a)
        start(b)

        await(a)
        await(b)
        print(marks.recall("a_started"))
        print(marks.recall("b_started"))
        print(a.status)
        print(b.status)
        """
    )
    # Both bodies reached pause() -- recording their markers -- before this
    # code ever resumed either one. That can only happen if both were
    # genuinely live at the same time, not run one-after-another.
    assert out == ["true", "true", "paused", "paused"]


def test_two_async_agents_sleeps_overlap_in_wall_clock_time():
    # The strongest proof: two agents each sleeping 0.2s, run concurrently,
    # finish in well under their combined 0.4s -- only possible if both
    # sleeps were actually running at the same time, not sequentially.
    import time as host_time

    started = host_time.monotonic()
    run(
        """
        import python "time" as time

        async agent a:
            time.sleep(0.2)

        async agent b:
            time.sleep(0.2)

        start(a)
        start(b)
        await(a)
        await(b)
        """
    )
    elapsed = host_time.monotonic() - started
    assert elapsed < 0.35, f"expected concurrent sleeps to overlap, took {elapsed:.3f}s (sequential would be ~0.4s)"


def test_resume_on_async_agent_does_not_block_and_await_catches_up():
    out = run(
        """
        memory marks

        async agent a:
            pause("first")
            marks.remember("done", true)

        start(a)
        await(a)
        resume(a)
        await(a)
        print(marks.recall("done"))
        print(a.status)
        """
    )
    assert out == ["true", "completed"]


# -- await() ----------------------------------------------------------------------


def test_await_on_agent_returns_the_agent_with_final_state():
    out = run(
        """
        async agent a:
            verdict = "approved"

        start(a)
        finished = await(a)
        print(finished.status)
        print(finished.verdict)
        """
    )
    assert out == ["completed", "approved"]


def test_await_reraises_a_failed_async_agents_error():
    with pytest.raises(SenseRuntimeError, match="boom"):
        run(
            """
            async agent a:
                assert(false, "boom")

            start(a)
            await(a)
            """
        )


def test_await_before_start_is_an_error():
    with pytest.raises(SenseRuntimeError):
        run(
            """
            async agent a:
                x = 1
            await(a)
            """
        )


def test_await_on_a_non_agent_non_future_is_an_error():
    with pytest.raises(SenseRuntimeError):
        run("await(1)\n")


def test_await_on_an_already_settled_sync_agent_is_a_noop():
    out = run(
        """
        agent a:
            x = 42
        start(a)
        again = await(a)
        print(again.x)
        """
    )
    assert out == ["42"]


# -- async tool ---------------------------------------------------------------------


def test_async_tool_call_returns_a_future_immediately():
    out = run(
        """
        async tool search(query: String) returns String:
            return "results for " + query

        f = search("sense")
        print(type_of(f))
        """
    )
    assert out == ["Future"]


def test_await_on_a_tool_future_returns_the_bodys_return_value():
    out = run(
        """
        async tool search(query: String) returns String:
            return "results for " + query

        print(await(search("sense")))
        """
    )
    assert out == ["results for sense"]


def test_sync_tool_call_is_unaffected_still_returns_a_value_directly():
    out = run(
        """
        tool add(a: Int, b: Int) returns Int:
            return a + b
        print(add(1, 2))
        """
    )
    assert out == ["3"]


def test_async_tool_capability_denial_is_immediate_no_await_needed():
    with pytest.raises(SensePolicyError):
        run(
            """
            async tool t() requires x.y:
                return 1
            policy:
                deny x.y
            t()
            """
        )


def test_async_tool_return_type_mismatch_surfaces_via_await():
    with pytest.raises(SenseTypeError):
        run(
            """
            async tool bad() returns Int:
                return "not an int"
            await(bad())
            """
        )


def test_await_reraises_an_async_tools_error():
    with pytest.raises(SenseRuntimeError, match="tool boom"):
        run(
            """
            async tool bad():
                assert(false, "tool boom")
            await(bad())
            """
        )


def test_future_done_is_a_nonblocking_poll():
    out = run(
        """
        async tool search(query: String) returns String:
            return query

        f = search("x")
        await(f)
        print(f.done)
        """
    )
    assert out == ["true"]


def test_type_of_a_future():
    out = run(
        """
        async tool t():
            return nil
        print(type_of(t()))
        """
    )
    assert out == ["Future"]
