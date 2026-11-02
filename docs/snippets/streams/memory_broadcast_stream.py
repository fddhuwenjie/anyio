from anyio import create_memory_broadcast_stream, create_task_group, run
from anyio.streams.memory import MemoryBroadcastReceiveStream


async def subscriber(
    name: str, receive_stream: MemoryBroadcastReceiveStream[str]
) -> None:
    async with receive_stream:
        async for event in receive_stream:
            print(f"{name} handled {event!r}")


async def main() -> None:
    send_stream, receive_stream = create_memory_broadcast_stream[str]()
    async with create_task_group() as tg, send_stream:
        # The original receive stream subscribes the first task, and every
        # clone subscribes another independent observer.
        tg.start_soon(subscriber, "audit", receive_stream)
        tg.start_soon(subscriber, "metrics", receive_stream.clone())
        tg.start_soon(subscriber, "handler", receive_stream.clone())

        # Each event is delivered to all three subscribers, in send order.
        await send_stream.send("user_logged_in")
        await send_stream.send("record_deleted")


run(main)
