from __future__ import annotations

import gc
import math
from typing import TYPE_CHECKING, NoReturn

import pytest

from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    create_memory_broadcast_stream,
    create_task_group,
    wait_all_tasks_blocked,
)

from ..conftest import asyncio_params

if TYPE_CHECKING:
    from anyio.abc import ObjectReceiveStream, ObjectSendStream, TaskStatus
    from anyio.streams.memory import MemoryBroadcastReceiveStream


def test_invalid_max_buffer() -> None:
    pytest.raises(ValueError, create_memory_broadcast_stream, 1.0).match(
        "max_buffer_size must be either an integer or math.inf"
    )


def test_negative_max_buffer() -> None:
    pytest.raises(ValueError, create_memory_broadcast_stream, -1).match(
        "max_buffer_size cannot be negative"
    )


async def test_broadcast_to_multiple_subscribers() -> None:
    async def reader(
        receive: MemoryBroadcastReceiveStream[str], out: list[str]
    ) -> None:
        out.extend([item async for item in receive])

    send, receive1 = create_memory_broadcast_stream[str](0)
    receive2 = receive1.clone()
    receive3 = receive1.clone()
    received1: list[str] = []
    received2: list[str] = []
    received3: list[str] = []
    async with send, create_task_group() as tg:
        tg.start_soon(reader, receive1, received1)
        tg.start_soon(reader, receive2, received2)
        tg.start_soon(reader, receive3, received3)
        await wait_all_tasks_blocked()
        await send.send("hello")
        await send.send("anyio")
        await send.aclose()

    assert received1 == ["hello", "anyio"]
    assert received2 == ["hello", "anyio"]
    assert received3 == ["hello", "anyio"]
    await receive1.aclose()
    await receive2.aclose()
    await receive3.aclose()


async def test_clone_subscribes_from_clone_time() -> None:
    send, receive1 = create_memory_broadcast_stream[str](math.inf)
    send.send_nowait("before")
    receive2 = receive1.clone()
    send.send_nowait("after")
    await send.aclose()

    assert [item async for item in receive1] == ["before", "after"]
    assert [item async for item in receive2] == ["after"]
    await receive1.aclose()
    await receive2.aclose()


async def test_buffered_clone_does_not_receive_history() -> None:
    send, receive1 = create_memory_broadcast_stream[str](3)
    send.send_nowait("one")
    send.send_nowait("two")
    receive2 = receive1.clone()
    send.send_nowait("three")
    await send.aclose()

    assert receive1.receive_nowait() == "one"
    assert receive1.receive_nowait() == "two"
    assert receive1.receive_nowait() == "three"
    with pytest.raises(EndOfStream):
        receive1.receive_nowait()

    assert receive2.receive_nowait() == "three"
    with pytest.raises(EndOfStream):
        receive2.receive_nowait()
    await receive1.aclose()
    await receive2.aclose()


async def test_zero_capacity_requires_every_subscriber() -> None:
    send, receive1 = create_memory_broadcast_stream[str](0)
    receive2 = receive1.clone()
    received: list[str] = []

    async def reader(receive: MemoryBroadcastReceiveStream[str]) -> None:
        received.append(await receive.receive())

    async with send, receive1, receive2, create_task_group() as tg:
        tg.start_soon(reader, receive1)
        await wait_all_tasks_blocked()
        tg.start_soon(send.send, "hello")
        await wait_all_tasks_blocked()
        assert received == []

        # The send can only complete once the second subscriber receives too.
        tg.start_soon(reader, receive2)
        await wait_all_tasks_blocked()

    assert sorted(received) == ["hello", "hello"]


async def test_send_nowait_would_block() -> None:
    send, receive1 = create_memory_broadcast_stream[str](0)
    receive2 = receive1.clone()

    async def receiver(
        receive: MemoryBroadcastReceiveStream[str],
        out: list[str],
        task_status: TaskStatus[None],
    ) -> None:
        task_status.started()
        out.append(await receive.receive())

    with send, receive1, receive2:
        with pytest.raises(WouldBlock):
            send.send_nowait("hello")

        received1: list[str] = []
        received2: list[str] = []
        async with create_task_group() as tg:
            await tg.start(receiver, receive1, received1)
            await tg.start(receiver, receive2, received2)
            await wait_all_tasks_blocked()
            send.send_nowait("hello")

        assert received1 == received2 == ["hello"]


async def test_send_nowait_behind_queued_send() -> None:
    send, receive1 = create_memory_broadcast_stream[str](0)
    receive2 = receive1.clone()
    async with send, receive1, receive2, create_task_group() as tg:
        tg.start_soon(send.send, "first")
        await wait_all_tasks_blocked()
        with pytest.raises(WouldBlock):
            send.send_nowait("second")

        tg.cancel_scope.cancel()


async def test_buffered_backpressure_from_slow_subscriber() -> None:
    send, fast = create_memory_broadcast_stream[str](1)
    slow = fast.clone()
    with send, fast, slow:
        send.send_nowait("m1")

        send_completed = False

        async def do_send() -> None:
            nonlocal send_completed
            await send.send("m2")
            send_completed = True

        async with create_task_group() as tg:
            tg.start_soon(do_send)
            await wait_all_tasks_blocked()
            assert not send_completed

            # The fast subscriber freeing a slot is not enough while the slow
            # one is still full.
            assert fast.receive_nowait() == "m1"
            await wait_all_tasks_blocked()
            assert not send_completed

            assert slow.receive_nowait() == "m1"
            await wait_all_tasks_blocked()
            assert send_completed
            assert fast.receive_nowait() == "m2"
            assert slow.receive_nowait() == "m2"


async def test_close_subscriber_releases_backpressure() -> None:
    send, receive1 = create_memory_broadcast_stream[str](1)
    receive2 = receive1.clone()
    send.send_nowait("m1")

    send_completed = False

    async def do_send() -> None:
        nonlocal send_completed
        await send.send("m2")
        send_completed = True

    async with send, receive1, create_task_group() as tg:
        tg.start_soon(do_send)
        await wait_all_tasks_blocked()
        assert not send_completed

        # Drain the remaining subscriber first; the send is still held back by
        # the closed subscriber's full buffer.
        assert receive1.receive_nowait() == "m1"
        await wait_all_tasks_blocked()
        assert not send_completed

        # Closing the slow (full) subscriber removes its backpressure.
        await receive2.aclose()
        await wait_all_tasks_blocked()
        assert send_completed

        assert receive1.receive_nowait() == "m2"


async def test_zero_capacity_close_unready_subscriber() -> None:
    send, receive1 = create_memory_broadcast_stream[str](0)
    receive2 = receive1.clone()
    send_completed = False

    async def do_send() -> None:
        nonlocal send_completed
        await send.send("hello")
        send_completed = True

    async with send, receive1, create_task_group() as tg:
        tg.start_soon(do_send)
        await wait_all_tasks_blocked()
        tg.start_soon(receive1.receive)
        await wait_all_tasks_blocked()
        assert not send_completed

        # The second subscriber never receives; closing it lets the send go
        # through to the remaining one.
        await receive2.aclose()
        await wait_all_tasks_blocked()
        assert send_completed


async def test_close_last_subscriber_while_sending() -> None:
    send, receive = create_memory_broadcast_stream[None](0)
    with send, pytest.RaisesGroup(BrokenResourceError):
        async with create_task_group() as tg:
            tg.start_soon(send.send, None)
            await wait_all_tasks_blocked()
            await receive.aclose()


async def test_send_after_all_subscribers_closed() -> None:
    send, receive = create_memory_broadcast_stream[None]()
    await receive.aclose()
    with send:
        with pytest.raises(BrokenResourceError):
            send.send_nowait(None)

        with pytest.raises(BrokenResourceError):
            await send.send(None)


async def test_drain_after_send_streams_closed() -> None:
    send, receive1 = create_memory_broadcast_stream[str](1)
    receive2 = receive1.clone()
    send.send_nowait("hello")
    await send.aclose()

    assert await receive1.receive() == "hello"
    assert await receive2.receive() == "hello"
    with pytest.raises(EndOfStream):
        await receive1.receive()
    with pytest.raises(EndOfStream):
        await receive2.receive()
    await receive1.aclose()
    await receive2.aclose()


async def test_receive_nowait_end_of_stream() -> None:
    send, receive = create_memory_broadcast_stream[None](1)
    await send.aclose()
    with receive:
        with pytest.raises(EndOfStream):
            receive.receive_nowait()


async def test_close_one_send_clone_while_sending() -> None:
    send1, receive = create_memory_broadcast_stream[str](0)
    send2 = send1.clone()
    with send2, receive:
        with pytest.RaisesGroup(ClosedResourceError):
            async with create_task_group() as tg:
                tg.start_soon(send1.send, "withdrawn")
                await wait_all_tasks_blocked()
                send1.close()

        async with create_task_group() as tg:
            tg.start_soon(send2.send, "hello")
            assert await receive.receive() == "hello"


async def test_cancel_send_withdraws_item() -> None:
    send, receive1 = create_memory_broadcast_stream[str](0)
    receive2 = receive1.clone()

    async def scoped_send(task_status: TaskStatus[CancelScope]) -> None:
        with CancelScope() as cancel_scope:
            task_status.started(cancel_scope)
            await send.send("hello")

    async def receiver(
        receive: MemoryBroadcastReceiveStream[str],
        out: list[str],
        task_status: TaskStatus[None],
    ) -> None:
        task_status.started()
        out.append(await receive.receive())

    with send, receive1, receive2:
        received1: list[str] = []
        received2: list[str] = []
        async with create_task_group() as tg:
            cancel_scope = await tg.start(scoped_send)
            await wait_all_tasks_blocked()
            cancel_scope.cancel()
            await wait_all_tasks_blocked()

            # The cancelled item was not delivered to anyone.
            with pytest.raises(WouldBlock):
                receive1.receive_nowait()
            with pytest.raises(WouldBlock):
                receive2.receive_nowait()

            await tg.start(receiver, receive1, received1)
            await tg.start(receiver, receive2, received2)
            await wait_all_tasks_blocked()
            await send.send("world")

        assert received1 == received2 == ["world"]


async def test_receive_nowait_handoff() -> None:
    send, receive1 = create_memory_broadcast_stream[str](0)
    receive2 = receive1.clone()
    send_completed = False

    async def do_send() -> None:
        nonlocal send_completed
        await send.send("hello")
        send_completed = True

    async with send, create_task_group() as tg:
        tg.start_soon(do_send)
        await wait_all_tasks_blocked()
        tg.start_soon(receive2.receive)
        await wait_all_tasks_blocked()
        assert receive1.receive_nowait() == "hello"
        await wait_all_tasks_blocked()
        assert send_completed

    await receive1.aclose()
    await receive2.aclose()


async def test_queued_sends_deliver_in_order() -> None:
    send, receive = create_memory_broadcast_stream[str](1)
    with send, receive:
        send.send_nowait("a")
        async with create_task_group() as tg:
            tg.start_soon(send.send, "b")
            await wait_all_tasks_blocked()
            tg.start_soon(send.send, "c")
            await wait_all_tasks_blocked()

            assert receive.receive_nowait() == "a"
            await wait_all_tasks_blocked()
            assert receive.receive_nowait() == "b"
            await wait_all_tasks_blocked()
            assert receive.receive_nowait() == "c"


async def test_close_receive_while_receiving() -> None:
    send, receive = create_memory_broadcast_stream[NoReturn]()
    with send, pytest.RaisesGroup(ClosedResourceError):
        async with create_task_group() as tg:
            tg.start_soon(receive.receive)
            await wait_all_tasks_blocked()
            await receive.aclose()


async def test_clone_closed() -> None:
    send, receive = create_memory_broadcast_stream[NoReturn]()
    await send.aclose()
    await receive.aclose()
    pytest.raises(ClosedResourceError, send.clone)
    pytest.raises(ClosedResourceError, receive.clone)


async def test_use_after_close() -> None:
    send, receive = create_memory_broadcast_stream[None]()
    send.close()
    receive.close()
    with pytest.raises(ClosedResourceError):
        send.send_nowait(None)
    with pytest.raises(ClosedResourceError):
        receive.receive_nowait()


async def test_statistics() -> None:
    send, receive1 = create_memory_broadcast_stream[None](1)
    receive2 = receive1.clone()
    with send, receive1, receive2:
        statistics = receive1.statistics()
        assert statistics.max_buffer_size == 1
        assert statistics.open_send_streams == 1
        assert statistics.open_receive_streams == 2
        assert statistics.tasks_waiting_send == 0
        assert statistics.tasks_waiting_receive == 0
        assert statistics.buffer_sizes == (0, 0)

        send.send_nowait(None)
        assert send.statistics().buffer_sizes == (1, 1)

        async with create_task_group() as tg:
            tg.start_soon(send.send, None)
            await wait_all_tasks_blocked()
            statistics = send.statistics()
            assert statistics.tasks_waiting_send == 1
            assert statistics.buffer_sizes == (1, 1)

            tg.start_soon(receive1.receive)
            tg.start_soon(receive2.receive)
            await wait_all_tasks_blocked()
            statistics = receive1.statistics()
            assert statistics.tasks_waiting_send == 0
            assert statistics.tasks_waiting_receive == 0
            assert statistics.buffer_sizes == (1, 1)

            assert receive1.receive_nowait() is None
            assert receive2.receive_nowait() is None
            assert send.statistics().buffer_sizes == (0, 0)

            await receive1.aclose()
            assert receive2.statistics().open_receive_streams == 1


async def test_type_variance() -> None:
    send, receive = create_memory_broadcast_stream[float]()
    with send, receive:
        receive1: ObjectReceiveStream[complex] = receive  # noqa: F841
        send1: ObjectSendStream[int] = send  # noqa: F841


async def test_not_closed_warning() -> None:
    send, receive = create_memory_broadcast_stream[int]()

    with pytest.warns(
        ResourceWarning,
        match="Unclosed <MemoryBroadcastSendStream at [0-9a-f]+>",
    ):
        del send
        gc.collect()

    with pytest.warns(
        ResourceWarning,
        match="Unclosed <MemoryBroadcastReceiveStream at [0-9a-f]+>",
    ):
        del receive
        gc.collect()


@pytest.mark.parametrize("anyio_backend", asyncio_params)
async def test_send_to_natively_cancelled_receiver() -> None:
    """
    A subscriber whose waiting task has a pending native cancellation must not
    count as accepting the item.

    """
    from asyncio import CancelledError, create_task

    send, receive1 = create_memory_broadcast_stream[str](0)
    receive2 = receive1.clone()

    async def receiver(receive: MemoryBroadcastReceiveStream[str]) -> None:
        received.append(await receive.receive())

    received: list[str] = []
    with send, receive1, receive2:
        receive_task = create_task(receive1.receive())
        await wait_all_tasks_blocked()
        receive_task.cancel()

        # The cancelled receiver does not accept, and the other subscriber is
        # not receiving either, so the send cannot go through.
        with pytest.raises(WouldBlock):
            send.send_nowait("hello")

        with pytest.raises(CancelledError):
            await receive_task

        with pytest.raises(WouldBlock):
            send.send_nowait("hello")

        async with create_task_group() as tg:
            tg.start_soon(receiver, receive1)
            tg.start_soon(receiver, receive2)
            await wait_all_tasks_blocked()
            await send.send("hello")

        assert sorted(received) == ["hello", "hello"]
