"""Opt-in two-slot HTTP owner for the generated KV-only batch decoder.

One thread owns all native prefill/decode/reset work. Request threads only
receive their own token events and invoke the existing response formatter.
Prefill is still synchronous; the prompt cap bounds, but does not interrupt,
a long generated prefill operation.
"""

from __future__ import annotations

import codecs
import queue
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any

from .batch_decode import Batch2RequestLoop
from .session_v8 import (
    CK_SESSION_REQUEST_RAW_PROMPT, SessionBusyError, SessionError,
    _longest_marker_prefix_len, truncate_stop_markers,
)


@dataclass
class _Client:
    slot: int
    events: queue.Queue = field(default_factory=queue.Queue)
    cancelled: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    delivered: threading.Event = field(default_factory=threading.Event)
    native_done: threading.Event = field(default_factory=threading.Event)
    ticket: int | None = None
    started: bool = False
    worker_started: bool = False
    terminal_sent: bool = False


class _Proxy:
    def __init__(self, owner: "Batch2HTTPOwner", client: _Client):
        self.owner = owner
        self.client = client

    def cancel(self):
        self.owner.cancel(self.client)

    def generate(self, _system, prompt, *, max_tokens, temperature, top_p,
                 on_token, flags=0, stop_on_text=(), stop_at_eos=False):
        try:
            return self._generate(
                _system, prompt, max_tokens=max_tokens, temperature=temperature,
                top_p=top_p, on_token=on_token, flags=flags,
                stop_on_text=stop_on_text, stop_at_eos=stop_at_eos)
        finally:
            self.client.finished.set()

    def _generate(self, _system, prompt, *, max_tokens, temperature, top_p,
                  on_token, flags=0, stop_on_text=(), stop_at_eos=False):
        if not (flags & CK_SESSION_REQUEST_RAW_PROMPT):
            raise ValueError("batch HTTP requires an already rendered raw prompt")
        ids = self.owner.session.encode_ids(prompt)
        with self.owner.guard:
            if self.owner.poisoned:
                raise SessionError(-7, "batch HTTP owner failed while preparing the prompt")
        if not ids or len(ids) > self.owner.max_prompt_tokens:
            raise ValueError("batch HTTP prompt exceeds its configured token cap")
        if len(ids) + max_tokens - 1 > self.owner.context_length:
            raise ValueError("batch HTTP prompt and output exceed context capacity")
        if max_tokens > self.owner.max_output_tokens:
            raise ValueError("batch HTTP output exceeds its configured delivery cap")
        with self.owner.guard:
            if self.owner.poisoned:
                raise SessionError(-7, "batch HTTP owner is unavailable after native failure")
            if self.owner.shutdown.is_set() or self.client.cancelled.is_set():
                return {"prompt_tokens": len(ids), "generated_tokens": 0,
                        "stop_reason": 3, "timing_unavailable": True}
            self.client.started = True
            self.owner.pending.put((self.client, ids, max_tokens, temperature, top_p))
        output_ids: list[int] = []
        visible_bytes = b""
        utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        stop_markers = [str(m) for m in stop_on_text if str(m)]
        if stop_at_eos:
            stop_markers.append("<eos>")
        generated_text = ""
        visible_chars = 0
        stop_text_hit = False

        def publish(delta: str, token_id: int):
            nonlocal generated_text, visible_chars, stop_text_hit
            generated_text += delta
            truncated = truncate_stop_markers(generated_text, stop_markers)
            hit = len(truncated) < len(generated_text)
            if hit:
                target = len(truncated)
            else:
                hold = _longest_marker_prefix_len(generated_text, stop_markers)
                target = len(generated_text) - hold
            if target > visible_chars:
                if on_token(token_id, generated_text[visible_chars:target]):
                    self.cancel()
            visible_chars = max(visible_chars, target)
            if hit:
                stop_text_hit = True
                self.cancel()

        while True:
            kind, value = self.client.events.get()
            if kind == "token":
                if self.client.cancelled.is_set():
                    continue
                output_ids.append(value)
                decoded = self.owner.session.decode_ids(output_ids)
                if len(decoded) > self.owner.max_output_bytes:
                    self.cancel()
                    raise SessionError(-7, "batch HTTP response exceeded its delivery byte cap")
                if not decoded.startswith(visible_bytes):
                    self.cancel()
                    raise SessionError(-7, "batch detokenization changed already emitted text")
                delta = utf8.decode(decoded[len(visible_bytes):], final=False)
                visible_bytes = decoded
                publish(delta, value)
            elif kind == "done":
                if value == "runtime_error":
                    raise SessionError(-7, "native batch step failed; active slots were reset")
                if not self.client.cancelled.is_set():
                    tail = utf8.decode(b"", final=True)
                    if tail:
                        publish(tail, output_ids[-1] if output_ids else -1)
                    final_text = truncate_stop_markers(generated_text, stop_markers)
                    if len(final_text) > visible_chars:
                        on_token(output_ids[-1] if output_ids else -1,
                                 final_text[visible_chars:])
                return {
                    "prompt_tokens": len(ids),
                    "generated_tokens": len(output_ids),
                    "stop_reason": (1 if stop_text_hit else
                                    {"eos": 1, "token_limit": 2, "cancelled": 3}.get(value, 0)),
                    "timing_unavailable": True,
                }


class _Lease:
    def __init__(self, owner: "Batch2HTTPOwner", client: _Client):
        self.owner = owner
        self.client = client
        self.session = _Proxy(owner, client)
        self.guard = threading.Lock()
        self.active = True
        self.worker_started = False
        self.cancelled = None

    def start_worker(self, cancelled):
        with self.guard:
            if not self.active or self.owner.shutdown.is_set():
                return False
            self.worker_started = True
            self.client.worker_started = True
            self.cancelled = cancelled
            return True

    def release(self):
        with self.guard:
            if self.active:
                self.active = False
                self.client.finished.set()
                self.owner.release(self.client)

    def delivery_done(self):
        self.owner.delivery_done(self.client)

    def cancel(self):
        with self.guard:
            if self.active:
                if self.cancelled is not None:
                    self.cancelled.set()
                self.session.cancel()

    def disconnect(self):
        self.cancel()
        with self.guard:
            if not self.worker_started and self.active:
                self.active = False
                self.client.finished.set()
                self.owner.release(self.client)


class Batch2HTTPOwner:
    """At most two admitted HTTP requests; one native execution thread."""

    def __init__(self, session, *, max_extra_bytes: int,
                 max_prompt_tokens: int, stop_ids=(), loop=None,
                 max_delivery_consumers: int = 4,
                 max_output_tokens: int = 4096,
                 max_output_bytes: int = 1 << 20,
                 delivery_timeout: float = 30.0):
        if max_prompt_tokens <= 0 or session.context_length is None:
            raise ValueError("batch HTTP needs positive prompt and context limits")
        if min(max_delivery_consumers, max_output_tokens, max_output_bytes) <= 0 or delivery_timeout <= 0:
            raise ValueError("batch HTTP delivery limits must be positive")
        self.session = session
        self.context_length = session.context_length
        self.max_prompt_tokens = max_prompt_tokens
        self.max_delivery_consumers = max_delivery_consumers
        self.max_output_tokens = max_output_tokens
        self.max_output_bytes = max_output_bytes
        self.delivery_timeout = delivery_timeout
        self.stop_ids = tuple(stop_ids)
        self.loop = loop if loop is not None else Batch2RequestLoop(
            session.enable_batch2(max_extra_bytes))
        self.pending: queue.Queue = queue.Queue()
        self.guard = threading.Lock()
        self.slots: dict[int, _Client] = {}
        self.clients: dict[int, _Client] = {}
        self.tickets: dict[int, _Client] = {}
        self.poisoned = False
        self.shutdown = threading.Event()
        self.shared_steps = 0
        self.worker = threading.Thread(target=self._run, name="cke-batch2-http", daemon=True)
        self.worker.start()

    def reserve(self):
        with self.guard:
            if self.shutdown.is_set():
                raise SessionError(-6, "batch HTTP owner is shutting down")
            if self.poisoned:
                raise SessionError(-7, "batch HTTP owner is unavailable after native failure")
            if len(self.clients) >= self.max_delivery_consumers:
                raise SessionBusyError(-6, "batch HTTP delivery consumers are full")
            for slot in (0, 1):
                if slot not in self.slots:
                    client = _Client(slot)
                    self.slots[slot] = client
                    self.clients[id(client)] = client
                    return _Lease(self, client)
        raise SessionBusyError(-6, "batch HTTP slots are full")

    def cancel(self, client: _Client):
        client.cancelled.set()
        with self.guard:
            ticket = client.ticket
            poisoned = self.poisoned
        if ticket is not None and not poisoned:
            self.loop.request_cancel(ticket)

    def release(self, client: _Client):
        with self.guard:
            if not client.started and self.slots.get(client.slot) is client:
                del self.slots[client.slot]
                client.native_done.set()
            if not client.worker_started or (client.delivered.is_set() and
                                             client.native_done.is_set()):
                self.clients.pop(id(client), None)
        if client.started:
            self.cancel(client)

    def delivery_done(self, client: _Client):
        client.delivered.set()
        with self.guard:
            if client.native_done.is_set():
                self.clients.pop(id(client), None)

    def _send_terminal(self, client: _Client, reason: str):
        with self.guard:
            if client.terminal_sent:
                return
            client.terminal_sent = True
        client.events.put(("done", reason))

    def _finish(self, ticket: int, reason: str):
        with self.guard:
            client = self.tickets.pop(ticket, None)
            if client is not None and self.slots.get(client.slot) is client:
                del self.slots[client.slot]
            if client is not None:
                client.native_done.set()
                if client.delivered.is_set():
                    self.clients.pop(id(client), None)
        if client is not None:
            self._send_terminal(client, reason)

    def _poison_all(self):
        # A native exception may invalidate every slot, including a request
        # pending prefill. Preparing consumers observe poison after encoding.
        with self.guard:
            self.poisoned = True
            admitted = tuple(self.clients.values())
            self.tickets.clear()
            self.slots.clear()
            while True:
                try:
                    self.pending.get_nowait()
                except queue.Empty:
                    break
            for client in admitted:
                client.native_done.set()
                if client.delivered.is_set():
                    self.clients.pop(id(client), None)
        for client in admitted:
            if client.worker_started:
                self._send_terminal(client, "runtime_error")

    def _run(self):
        try:
            self._run_owned()
        except Exception:
            self._poison_all()

    def _run_owned(self):
        while True:
            if self.shutdown.is_set() and not self.loop.active_tickets() and self.pending.empty():
                return
            # Admit one pending request per iteration, then advance both
            # active rows through the same generated native step.
            try:
                client, ids, max_tokens, temperature, top_p = self.pending.get(
                    timeout=0.05 if not self.loop.active_tickets() else 0)
            except queue.Empty:
                client = None
            if client is not None:
                if client.cancelled.is_set():
                    with self.guard:
                        if self.slots.get(client.slot) is client:
                            del self.slots[client.slot]
                        client.native_done.set()
                        if client.delivered.is_set():
                            self.clients.pop(id(client), None)
                    self._send_terminal(client, "cancelled")
                else:
                    ticket = self.loop.submit(
                        client.slot, ids, max_tokens=max_tokens,
                        stop_ids=self.stop_ids, temperature=temperature,
                        top_p=top_p, seed=secrets.randbits(63))
                    with self.guard:
                        client.ticket = ticket
                        self.tickets[ticket] = client
                    if client.cancelled.is_set():
                        self.loop.request_cancel(ticket)
            if not self.loop.active_tickets():
                continue
            tick = self.loop.advance()
            for event in tick.tokens:
                with self.guard:
                    recipient = self.tickets.get(event.ticket)
                if recipient is not None:
                    if not recipient.terminal_sent:
                        recipient.events.put(("token", event.token_id))
            for event in tick.completed:
                self._finish(event.ticket, event.reason)
            if tick.advanced_mask == 3:
                self.shared_steps += 1

    def close(self, timeout: float = 10.0):
        """Join the native owner before its session can be closed."""
        with self.guard:
            self.shutdown.set()
            clients = tuple(self.clients.values())
        for client in clients:
            self.cancel(client)
        self.worker.join(timeout)
        if self.worker.is_alive():
            raise SessionError(-7, "native batch owner did not stop before shutdown deadline")
        for client in clients:
            if client.worker_started and not client.finished.wait(timeout):
                raise SessionError(-7, "HTTP batch consumer still uses native tokenizer at shutdown")
