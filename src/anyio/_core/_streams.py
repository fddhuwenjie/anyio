from __future__ import annotations

import math
from typing import TypeVar
from warnings import warn

from ..streams.memory import (
    MemoryBroadcastReceiveStream,
    MemoryBroadcastSendStream,
    MemoryObjectReceiveStream,
    MemoryObjectSendStream,
    _MemoryBroadcastStreamState,
    _MemoryObjectStreamState,
)

T_Item = TypeVar("T_Item")


def _validate_max_buffer_size(max_buffer_size: float) -> None:
    if max_buffer_size != math.inf and not isinstance(max_buffer_size, int):
        raise ValueError("max_buffer_size must be either an integer or math.inf")
    if max_buffer_size < 0:
        raise ValueError("max_buffer_size cannot be negative")


class create_memory_object_stream(
    tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]],
):
    """
    Create a memory object stream.

    The stream's item type can be annotated like
    :func:`create_memory_object_stream[T_Item]`.

    :param max_buffer_size: number of items held in the buffer until ``send()`` starts
        blocking
    :param item_type: old way of marking the streams with the right generic type for
        static typing (does nothing on AnyIO 4)

        .. deprecated:: 4.0
          Use ``create_memory_object_stream[YourItemType](...)`` instead.
    :return: a tuple of (send stream, receive stream)

    """

    def __new__(  # type: ignore[misc]
        cls, max_buffer_size: float = 0, item_type: object = None
    ) -> tuple[MemoryObjectSendStream[T_Item], MemoryObjectReceiveStream[T_Item]]:
        _validate_max_buffer_size(max_buffer_size)
        if item_type is not None:
            warn(
                "The item_type argument has been deprecated in AnyIO 4.0. "
                "Use create_memory_object_stream[YourItemType](...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        state = _MemoryObjectStreamState[T_Item](max_buffer_size)
        return (MemoryObjectSendStream(state), MemoryObjectReceiveStream(state))


class create_memory_broadcast_stream(
    tuple[MemoryBroadcastSendStream[T_Item], MemoryBroadcastReceiveStream[T_Item]],
):
    """
    Create a memory broadcast stream.

    Unlike a memory object stream, every item sent through this stream is
    delivered to every open subscriber, not just one of them.

    The receive stream returned by this function acts as the first
    subscriber. Calling
    :meth:`~anyio.streams.memory.MemoryBroadcastReceiveStream.clone` on it
    (or on any of its clones) creates another independent subscriber that
    only receives items sent after the clone was created.

    ``send()`` (and ``send_nowait()``) only succeeds when every open
    subscriber can accept the item at the same time. An item is thus never
    delivered to only some of the subscribers. With a buffer size of 0, the
    sender blocks until every subscriber is waiting to receive the item.
    With a finite buffer size, the slowest subscriber exerts backpressure
    on the senders.

    Closing a receive stream (ending one subscription) releases the
    backpressure it was exerting without affecting the other subscribers.
    Once all subscribers have been closed, sends raise
    :exc:`~anyio.BrokenResourceError`. Once all clones of the send stream
    have been closed, each subscriber, after draining the items it has
    already received, receives an end of stream indication.

    The stream's item type can be annotated like
    :func:`create_memory_broadcast_stream[T_Item]`.

    :param max_buffer_size: number of items held in each subscriber's
        buffer until ``send()`` starts blocking (0 by default)
    :return: a tuple of (send stream, receive stream)

    """

    def __new__(  # type: ignore[misc]
        cls, max_buffer_size: float = 0
    ) -> tuple[MemoryBroadcastSendStream[T_Item], MemoryBroadcastReceiveStream[T_Item]]:
        _validate_max_buffer_size(max_buffer_size)
        state = _MemoryBroadcastStreamState[T_Item](max_buffer_size)
        return (MemoryBroadcastSendStream(state), MemoryBroadcastReceiveStream(state))
