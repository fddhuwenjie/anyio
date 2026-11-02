from __future__ import annotations

__all__ = (
    "MemoryBroadcastReceiveStream",
    "MemoryBroadcastSendStream",
    "MemoryBroadcastStreamStatistics",
    "MemoryObjectReceiveStream",
    "MemoryObjectSendStream",
    "MemoryObjectStreamStatistics",
)

import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from types import TracebackType
from typing import Generic, NamedTuple, TypeVar

from .._core._exceptions import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
)
from .._core._synchronization import Event
from .._core._testing import TaskInfo, get_current_task
from ..abc import ObjectReceiveStream, ObjectSendStream
from ..lowlevel import checkpoint

T_Item = TypeVar("T_Item")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


class MemoryObjectStreamStatistics(NamedTuple):
    current_buffer_used: int  #: number of items stored in the buffer
    #: maximum number of items that can be stored on this stream (or :data:`math.inf`)
    max_buffer_size: float
    open_send_streams: int  #: number of unclosed clones of the send stream
    open_receive_streams: int  #: number of unclosed clones of the receive stream
    #: number of tasks blocked on :meth:`MemoryObjectSendStream.send`
    tasks_waiting_send: int
    #: number of tasks blocked on :meth:`MemoryObjectReceiveStream.receive`
    tasks_waiting_receive: int


@dataclass(eq=False)
class _MemoryObjectItemReceiver(Generic[T_Item]):
    task_info: TaskInfo = field(init=False, default_factory=get_current_task)
    item: T_Item = field(init=False)

    def __repr__(self) -> str:
        # When item is not defined, we get following error with default __repr__:
        # AttributeError: 'MemoryObjectItemReceiver' object has no attribute 'item'
        item = getattr(self, "item", None)
        return f"{self.__class__.__name__}(task_info={self.task_info}, item={item!r})"


@dataclass(eq=False)
class _MemoryObjectStreamState(Generic[T_Item]):
    max_buffer_size: float = field()
    buffer: deque[T_Item] = field(init=False, default_factory=deque)
    open_send_channels: int = field(init=False, default=0)
    open_receive_channels: int = field(init=False, default=0)
    waiting_receivers: OrderedDict[Event, _MemoryObjectItemReceiver[T_Item]] = field(
        init=False, default_factory=OrderedDict
    )
    waiting_senders: OrderedDict[Event, T_Item] = field(
        init=False, default_factory=OrderedDict
    )

    def statistics(self) -> MemoryObjectStreamStatistics:
        return MemoryObjectStreamStatistics(
            len(self.buffer),
            self.max_buffer_size,
            self.open_send_channels,
            self.open_receive_channels,
            len(self.waiting_senders),
            len(self.waiting_receivers),
        )


@dataclass(eq=False)
class MemoryObjectReceiveStream(ObjectReceiveStream[T_co], Generic[T_co]):
    _state: _MemoryObjectStreamState[T_co]
    _closed: bool = field(init=False, default=False)
    _waiters: set[Event] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._state.open_receive_channels += 1

    def receive_nowait(self) -> T_co:
        """
        Receive the next item if it can be done without waiting.

        :return: the received item
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.EndOfStream: if the buffer is empty and this stream has been
            closed from the sending end
        :raises ~anyio.WouldBlock: if there are no items in the buffer and no tasks
            waiting to send

        """
        if self._closed:
            raise ClosedResourceError

        if self._state.waiting_senders:
            # Get the item from the next sender
            send_event, item = self._state.waiting_senders.popitem(last=False)
            self._state.buffer.append(item)
            send_event.set()

        if self._state.buffer:
            return self._state.buffer.popleft()
        elif not self._state.open_send_channels:
            raise EndOfStream

        raise WouldBlock

    async def receive(self) -> T_co:
        await checkpoint()
        try:
            return self.receive_nowait()
        except WouldBlock:
            if self._waiters is None:
                self._waiters = set[Event]()

            # Add ourselves in the queue
            receive_event = Event()
            self._waiters.add(receive_event)
            receiver = _MemoryObjectItemReceiver[T_co]()
            self._state.waiting_receivers[receive_event] = receiver

            try:
                await receive_event.wait()
            finally:
                self._state.waiting_receivers.pop(receive_event, None)
                if self._waiters is not None:
                    self._waiters.remove(receive_event)

            try:
                return receiver.item
            except AttributeError:
                if self._closed:
                    raise ClosedResourceError from None
                else:
                    raise EndOfStream from None

    def clone(self) -> MemoryObjectReceiveStream[T_co]:
        """
        Create a clone of this receive stream.

        Each clone can be closed separately. Only when all clones have been closed will
        the receiving end of the memory stream be considered closed by the sending ends.

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return MemoryObjectReceiveStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a special
        case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            self._state.open_receive_channels -= 1
            if self._state.open_receive_channels == 0:
                send_events = list(self._state.waiting_senders.keys())
                for event in send_events:
                    event.set()

            if self._waiters is not None:
                for event in self._waiters:
                    self._state.waiting_receivers.pop(event, None)
                    event.set()

                self._waiters = None

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> MemoryObjectStreamStatistics:
        """
        Return statistics about the current state of this stream.

        .. versionadded:: 3.0
        """
        return self._state.statistics()

    def __enter__(self) -> MemoryObjectReceiveStream[T_co]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )


@dataclass(eq=False)
class MemoryObjectSendStream(ObjectSendStream[T_contra], Generic[T_contra]):
    _state: _MemoryObjectStreamState[T_contra]
    _closed: bool = field(init=False, default=False)
    _waiters: set[Event] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._state.open_send_channels += 1

    def send_nowait(self, item: T_contra) -> None:
        """
        Send an item immediately if it can be done without waiting.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if the stream has been closed from the
            receiving end
        :raises ~anyio.WouldBlock: if the buffer is full and there are no tasks waiting
            to receive

        """
        if self._closed:
            raise ClosedResourceError
        if not self._state.open_receive_channels:
            raise BrokenResourceError

        while self._state.waiting_receivers:
            receive_event, receiver = self._state.waiting_receivers.popitem(last=False)
            if not receiver.task_info.has_pending_cancellation():
                receiver.item = item
                receive_event.set()
                return

        if len(self._state.buffer) < self._state.max_buffer_size:
            self._state.buffer.append(item)
        else:
            raise WouldBlock

    async def send(self, item: T_contra) -> None:
        """
        Send an item to the stream.

        If the buffer is full, this method blocks until there is again room in the
        buffer or the item can be sent directly to a receiver.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been closed
        :raises ~anyio.BrokenResourceError: if the stream has been closed from the
            receiving end

        """
        await checkpoint()
        try:
            self.send_nowait(item)
        except WouldBlock:
            if self._waiters is None:
                self._waiters = set[Event]()

            # Wait until there's someone on the receiving end
            send_event = Event()
            self._waiters.add(send_event)
            self._state.waiting_senders[send_event] = item
            try:
                await send_event.wait()
            finally:
                if item_not_consumed := send_event in self._state.waiting_senders:
                    del self._state.waiting_senders[send_event]

                withdrawn = self._closed and send_event in self._waiters
                self._waiters.discard(send_event)

            if item_not_consumed:
                raise BrokenResourceError from None
            elif withdrawn:
                raise ClosedResourceError from None

    def clone(self) -> MemoryObjectSendStream[T_contra]:
        """
        Create a clone of this send stream.

        Each clone can be closed separately. Only when all clones have been closed will
        the sending end of the memory stream be considered closed by the receiving ends.

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return MemoryObjectSendStream(_state=self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a special
        case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            self._state.open_send_channels -= 1
            if self._state.open_send_channels == 0:
                receive_events = list(self._state.waiting_receivers.keys())
                self._state.waiting_receivers.clear()
                for event in receive_events:
                    event.set()

            if self._waiters is not None:
                withdrawn = set[Event]()
                for event in self._waiters:
                    if event in self._state.waiting_senders:
                        del self._state.waiting_senders[event]
                        withdrawn.add(event)

                    event.set()

                # An event remaining on this set signals that its item was never
                # delivered, and send() will raise ClosedResourceError
                self._waiters = withdrawn

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> MemoryObjectStreamStatistics:
        """
        Return statistics about the current state of this stream.

        .. versionadded:: 3.0
        """
        return self._state.statistics()

    def __enter__(self) -> MemoryObjectSendStream[T_contra]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )


class MemoryBroadcastStreamStatistics(NamedTuple):
    #: maximum number of items that can be buffered for each subscriber
    max_buffer_size: float
    open_send_streams: int  #: number of unclosed clones of the send stream
    #: number of independent subscribers (unclosed clones of the receive stream)
    open_receive_streams: int
    #: number of tasks blocked on :meth:`MemoryBroadcastSendStream.send`
    tasks_waiting_send: int
    #: number of tasks blocked on :meth:`MemoryBroadcastReceiveStream.receive`
    tasks_waiting_receive: int
    #: number of items buffered for each subscriber, in subscription order
    buffer_sizes: tuple[int, ...]


@dataclass(eq=False)
class _BroadcastSubscription(Generic[T_Item]):
    buffer: deque[T_Item] = field(init=False, default_factory=deque)
    waiting_receivers: OrderedDict[Event, _MemoryObjectItemReceiver[T_Item]] = field(
        init=False, default_factory=OrderedDict
    )
    #: when set, a synchronous ``receive_nowait()`` call offers itself as the
    #: immediate handoff target for this subscription
    immediate_slot: list[T_Item] | None = field(init=False, default=None)


@dataclass(eq=False)
class _MemoryBroadcastStreamState(Generic[T_Item]):
    max_buffer_size: float = field()
    subscriptions: dict[_BroadcastSubscription[T_Item], None] = field(
        init=False, default_factory=dict
    )
    open_send_channels: int = field(init=False, default=0)
    open_receive_channels: int = field(init=False, default=0)
    waiting_senders: OrderedDict[Event, T_Item] = field(
        init=False, default_factory=OrderedDict
    )

    def statistics(self) -> MemoryBroadcastStreamStatistics:
        return MemoryBroadcastStreamStatistics(
            self.max_buffer_size,
            self.open_send_channels,
            self.open_receive_channels,
            len(self.waiting_senders),
            sum(
                len(subscription.waiting_receivers)
                for subscription in self.subscriptions
            ),
            tuple(len(subscription.buffer) for subscription in self.subscriptions),
        )

    def _collect_targets(
        self, item: T_Item
    ) -> list[tuple[_BroadcastSubscription[T_Item], str, Event | None]] | None:
        """
        Check that every open subscriber can accept the item, and collect the
        per-subscriber delivery targets.

        :return: the delivery targets, or ``None`` if at least one subscriber
            cannot accept the item right now (in which case no state is changed)

        """
        targets: list[tuple[_BroadcastSubscription[T_Item], str, Event | None]] = []
        for subscription in self.subscriptions:
            for receive_event, receiver in subscription.waiting_receivers.items():
                if not receiver.task_info.has_pending_cancellation():
                    targets.append((subscription, "receiver", receive_event))
                    break
            else:
                if subscription.immediate_slot is not None:
                    targets.append((subscription, "immediate", None))
                elif len(subscription.buffer) < self.max_buffer_size:
                    targets.append((subscription, "buffer", None))
                else:
                    return None

        return targets

    def _deliver(
        self,
        targets: list[tuple[_BroadcastSubscription[T_Item], str, Event | None]],
        item: T_Item,
    ) -> None:
        """Deliver the item to every collected target (atomic commit)."""
        for subscription, kind, receive_event in targets:
            if kind == "receiver":
                assert receive_event is not None
                receiver = subscription.waiting_receivers.pop(receive_event)
                receiver.item = item
                receive_event.set()
            elif kind == "immediate":
                assert subscription.immediate_slot is not None
                subscription.immediate_slot.append(item)
            else:
                subscription.buffer.append(item)

    def _try_deliver(self, item: T_Item) -> bool:
        """
        Try to deliver the item to every open subscriber at once.

        :return: ``True`` if the item was delivered to every subscriber,
            ``False`` if at least one subscriber could not accept it (no state
            is changed in that case)

        """
        if not self.subscriptions:
            return False

        targets = self._collect_targets(item)
        if targets is None:
            return False

        self._deliver(targets, item)
        return True

    def pump(self) -> None:
        """
        Deliver as many queued sender items as every subscriber can currently
        accept, in send order.

        """
        while self.subscriptions and self.waiting_senders:
            send_event, item = next(iter(self.waiting_senders.items()))
            targets = self._collect_targets(item)
            if targets is None:
                return

            del self.waiting_senders[send_event]
            self._deliver(targets, item)
            send_event.set()


@dataclass(eq=False)
class MemoryBroadcastReceiveStream(ObjectReceiveStream[T_co], Generic[T_co]):
    _state: _MemoryBroadcastStreamState[T_co]
    _subscription: _BroadcastSubscription[T_co] = field(init=False)
    _closed: bool = field(init=False, default=False)
    _waiters: set[Event] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._subscription = _BroadcastSubscription[T_co]()
        self._state.subscriptions[self._subscription] = None
        self._state.open_receive_channels += 1

    def receive_nowait(self) -> T_co:
        """
        Receive the next item if it can be done without waiting.

        Only items sent after this subscriber was created are ever received.

        :return: the received item
        :raises ~anyio.ClosedResourceError: if this receive stream has been
            closed
        :raises ~anyio.EndOfStream: if this subscriber's buffer is empty and
            all clones of the send stream have been closed
        :raises ~anyio.WouldBlock: if there are no items for this subscriber
            and no item can be delivered to every subscriber right now

        """
        if self._closed:
            raise ClosedResourceError

        subscription = self._subscription
        if subscription.buffer:
            item = subscription.buffer.popleft()

            # Consuming a buffered item may unblock a broadcast send.
            self._state.pump()
            return item

        if self._state.waiting_senders:
            # Offer ourselves as a direct handoff target so that a blocked
            # sender can deliver its item to every subscriber at once.
            immediate_slot: list[T_co] = []
            subscription.immediate_slot = immediate_slot
            try:
                self._state.pump()
            finally:
                subscription.immediate_slot = None

            if immediate_slot:
                return immediate_slot[0]

        if not self._state.open_send_channels:
            raise EndOfStream

        raise WouldBlock

    async def receive(self) -> T_co:
        await checkpoint()
        try:
            return self.receive_nowait()
        except WouldBlock:
            if self._waiters is None:
                self._waiters = set[Event]()

            # Register ourselves as a waiting receiver for this subscriber.
            receive_event = Event()
            receiver = _MemoryObjectItemReceiver[T_co]()
            self._waiters.add(receive_event)
            self._subscription.waiting_receivers[receive_event] = receiver

            try:
                # This subscriber may have been the last one holding back a
                # broadcast send.
                self._state.pump()
                await receive_event.wait()
            finally:
                self._subscription.waiting_receivers.pop(receive_event, None)
                if self._waiters is not None:
                    self._waiters.remove(receive_event)

            try:
                return receiver.item
            except AttributeError:
                if self._closed:
                    raise ClosedResourceError from None
                else:
                    raise EndOfStream from None

    def clone(self) -> MemoryBroadcastReceiveStream[T_co]:
        """
        Create a new independent subscriber to this broadcast stream.

        The clone only receives items that are sent after the clone has been
        created; it will never see items sent earlier. The original stream
        remains a subscriber and is otherwise unaffected by the clone.

        :return: the cloned stream, a new independent subscriber

        """
        if self._closed:
            raise ClosedResourceError

        return MemoryBroadcastReceiveStream(self._state)

    def close(self) -> None:
        """
        Close the stream, ending this subscription.

        Backpressure previously exerted by this subscriber is released,
        allowing blocked sends to proceed for the remaining subscribers. This
        works the exact same way as :meth:`aclose`, but is provided as a
        special case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            state = self._state
            subscription = self._subscription
            state.subscriptions.pop(subscription, None)
            state.open_receive_channels -= 1

            # Wake up the tasks blocked in receive() on this subscriber.
            receive_events = list(subscription.waiting_receivers)
            subscription.waiting_receivers.clear()
            if self._waiters is not None:
                self._waiters.clear()
                self._waiters = None

            for event in receive_events:
                event.set()

            # This subscriber no longer exerts backpressure, so queued items
            # may now be deliverable to the remaining subscribers.
            state.pump()

            # With no subscribers left, all pending sends must fail.
            if not state.subscriptions:
                for event in state.waiting_senders:
                    event.set()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> MemoryBroadcastStreamStatistics:
        """
        Return statistics about the current state of this stream.
        """
        return self._state.statistics()

    def __enter__(self) -> MemoryBroadcastReceiveStream[T_co]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )


@dataclass(eq=False)
class MemoryBroadcastSendStream(ObjectSendStream[T_contra], Generic[T_contra]):
    _state: _MemoryBroadcastStreamState[T_contra]
    _closed: bool = field(init=False, default=False)
    _waiters: set[Event] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._state.open_send_channels += 1

    def send_nowait(self, item: T_contra) -> None:
        """
        Send an item to every open subscriber immediately if it can be done
        without waiting.

        The item is delivered either to every open subscriber or to none of
        them.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been
            closed
        :raises ~anyio.BrokenResourceError: if all subscribers have been
            closed
        :raises ~anyio.WouldBlock: if at least one subscriber cannot accept
            the item right now, or if another send is still in progress

        """
        if self._closed:
            raise ClosedResourceError
        if not self._state.subscriptions:
            raise BrokenResourceError
        if self._state.waiting_senders:
            # Another send is still in progress, and it must finish first.
            raise WouldBlock
        if not self._state._try_deliver(item):
            raise WouldBlock

    async def send(self, item: T_contra) -> None:
        """
        Send an item to every open subscriber.

        This method blocks until every open subscriber can accept the item at
        the same time (either by receiving it directly or by buffering it).
        The item is never delivered to only some of the subscribers. A slow
        subscriber therefore exerts backpressure on all sends.

        :param item: the item to send
        :raises ~anyio.ClosedResourceError: if this send stream has been
            closed
        :raises ~anyio.BrokenResourceError: if all subscribers have been
            closed while waiting to send

        """
        await checkpoint()
        try:
            self.send_nowait(item)
            return
        except WouldBlock:
            pass

        if self._waiters is None:
            self._waiters = set[Event]()

        send_event = Event()
        self._waiters.add(send_event)
        self._state.waiting_senders[send_event] = item
        try:
            self._state.pump()
            if not send_event.is_set():
                try:
                    await send_event.wait()
                except BaseException:
                    # Withdraw the item without delivering it to anyone.
                    if send_event in self._state.waiting_senders:
                        del self._state.waiting_senders[send_event]

                        # A later send may now be at the head of the queue.
                        self._state.pump()

                    raise

            item_not_consumed = send_event in self._state.waiting_senders
            if item_not_consumed:
                del self._state.waiting_senders[send_event]

            withdrawn = (
                self._closed
                and self._waiters is not None
                and send_event in self._waiters
            )
        finally:
            if self._waiters is not None:
                self._waiters.discard(send_event)

        if item_not_consumed:
            raise BrokenResourceError from None
        elif withdrawn:
            raise ClosedResourceError from None

    def clone(self) -> MemoryBroadcastSendStream[T_contra]:
        """
        Create a clone of this send stream.

        Each clone can be closed separately. Only when all clones have been
        closed will the subscribers, after draining their buffers, receive an
        end of stream indication.

        :return: the cloned stream

        """
        if self._closed:
            raise ClosedResourceError

        return MemoryBroadcastSendStream(self._state)

    def close(self) -> None:
        """
        Close the stream.

        This works the exact same way as :meth:`aclose`, but is provided as a
        special case for the benefit of synchronous callbacks.

        """
        if not self._closed:
            self._closed = True
            self._state.open_send_channels -= 1
            if self._waiters is not None:
                withdrawn = set[Event]()
                for event in self._waiters:
                    if event in self._state.waiting_senders:
                        del self._state.waiting_senders[event]
                        withdrawn.add(event)

                    event.set()

                # An event remaining in this set signals that its item was
                # never delivered, and send() will raise ClosedResourceError.
                self._waiters = withdrawn

            # The withdrawn item may have been holding back later sends.
            self._state.pump()

            if self._state.open_send_channels == 0:
                for subscription in self._state.subscriptions:
                    receive_events = list(subscription.waiting_receivers)
                    subscription.waiting_receivers.clear()
                    for event in receive_events:
                        event.set()

    async def aclose(self) -> None:
        self.close()

    def statistics(self) -> MemoryBroadcastStreamStatistics:
        """
        Return statistics about the current state of this stream.
        """
        return self._state.statistics()

    def __enter__(self) -> MemoryBroadcastSendStream[T_contra]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed <{self.__class__.__name__} at {id(self):x}>",
                ResourceWarning,
                stacklevel=1,
                source=self,
            )
