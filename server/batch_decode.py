"""Two-request token scheduler over the opt-in native batch2 session.

Inputs are already rendered and tokenized. The caller owns HTTP admission,
output transport, and worker lifetime. Optional staged prefill replays prompt
tokens through generated decode in caller-bounded chunks. No route enables
that path by default.
"""

from __future__ import annotations

import ctypes
import math
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

from .session_v8 import Batch2SessionV8, SessionError


@dataclass(frozen=True)
class TokenEvent:
    ticket: int
    slot: int
    token_id: int


@dataclass(frozen=True)
class CompletionEvent:
    ticket: int
    slot: int
    reason: str
    generated_tokens: int


@dataclass(frozen=True)
class BatchTick:
    tokens: tuple[TokenEvent, ...]
    completed: tuple[CompletionEvent, ...]
    advanced_mask: int | None
    mixed_prefill_ticket: int | None = None
    mixed_prefill_consumed: int | None = None


@dataclass
class _Request:
    slot: int
    ticket: int
    logits: Any
    max_tokens: int
    stop_ids: frozenset[int]
    temperature: float
    top_p: float
    rng: random.Random
    generated: int = 0
    cancel_requested: threading.Event = field(default_factory=threading.Event)


@dataclass
class _PrefillRequest:
    slot: int
    ticket: int
    prompt_ids: tuple[int, ...]
    max_tokens: int
    stop_ids: frozenset[int]
    temperature: float
    top_p: float
    rng: random.Random
    offset: int = 0
    cancel_requested: threading.Event = field(default_factory=threading.Event)


class Batch2RequestLoop:
    """One serialized worker with two independent request policies."""

    def __init__(self, batch: Batch2SessionV8, *, prefill_phase_staged: bool = False):
        if type(prefill_phase_staged) is not bool:
            raise ValueError("prefill_phase_staged must be boolean")
        self.batch = batch
        self.prefill_phase_staged = prefill_phase_staged
        self._operations = threading.Lock()
        self._state = threading.Lock()
        self._slots: dict[int, _Request] = {}
        self._prefills: dict[int, _PrefillRequest] = {}
        self._tickets: dict[int, _Request | _PrefillRequest] = {}
        self._poisoned = False
        sampler = getattr(batch.parent.lib, "ck_sample_top_p_v8", None)
        if sampler is None:
            raise SessionError(-5, "loaded session library lacks the explicit sampler ABI")
        sampler.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                            ctypes.c_float, ctypes.c_float, ctypes.c_float]
        sampler.restype = ctypes.c_int
        self._sample = sampler

    def _validate(self, slot: int, prompt_ids: Sequence[int], max_tokens: int,
                  stop_ids: Sequence[int], temperature: float,
                  top_p: float, seed: int) -> None:
        if type(slot) is not int or slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if not prompt_ids:
            raise ValueError("prompt must contain at least one token")
        if any(type(token) is not int or token < 0 or token >= self.batch.vocab_size
               for token in prompt_ids):
            raise ValueError("prompt IDs must belong to the loaded vocabulary")
        if not math.isfinite(temperature) or temperature < 0 or not math.isfinite(top_p) or not 0 < top_p <= 1:
            raise ValueError("invalid sampling parameters")
        if any(type(token) is not int or token < 0 or token >= self.batch.vocab_size
               for token in stop_ids):
            raise ValueError("stop IDs must belong to the loaded vocabulary")
        context = self.batch.parent.context_length
        # Prefill predicts the first output token; only later outputs need
        # another decode position in the native cache.
        if context is not None and len(prompt_ids) + max_tokens - 1 > context:
            raise ValueError("prompt plus output budget exceeds loaded context")

    def _check_available(self, slot: int) -> None:
        with self._state:
            if self._poisoned:
                raise SessionError(-7, "batch request loop needs recovery")
            if slot in self._slots or slot in self._prefills:
                raise SessionError(-6, "batch slot already owns a request")

    def submit(self, slot: int, prompt_ids: Sequence[int], *, max_tokens: int,
               stop_ids: Sequence[int] = (), temperature: float = 0.0,
               top_p: float = 1.0, seed: int = 0) -> int:
        self._validate(slot, prompt_ids, max_tokens, stop_ids, temperature, top_p, seed)
        with self._operations:
            self._check_available(slot)
            try:
                ticket, logits = self.batch.prefill(slot, prompt_ids)
            except SessionError:
                self._fail_all_after_native_error()
                raise
            request = _Request(slot, ticket, logits, max_tokens,
                               frozenset(stop_ids), float(temperature),
                               float(top_p), random.Random(seed))
            with self._state:
                self._slots[slot] = request
                self._tickets[ticket] = request
            return ticket

    def begin_staged(self, slot: int, prompt_ids: Sequence[int], *, max_tokens: int,
                     stop_ids: Sequence[int] = (), temperature: float = 0.0,
                     top_p: float = 1.0, seed: int = 0) -> int:
        """Reserve a native ticket before any prompt token is computed."""
        self._validate(slot, prompt_ids, max_tokens, stop_ids, temperature, top_p, seed)
        with self._operations:
            self._check_available(slot)
            try:
                ticket = self.batch.begin_prefill(slot, len(prompt_ids))
            except SessionError:
                self._fail_all_after_native_error()
                raise
            request = _PrefillRequest(slot, ticket, tuple(prompt_ids), max_tokens,
                                      frozenset(stop_ids), float(temperature),
                                      float(top_p), random.Random(seed))
            with self._state:
                self._prefills[slot] = request
                self._tickets[ticket] = request
            return ticket

    def staged_tickets(self) -> tuple[int, ...]:
        with self._state:
            return tuple(r.ticket for _, r in sorted(self._prefills.items()))

    def advance_prefill(self, ticket: int, max_tokens: int) -> tuple[CompletionEvent, ...]:
        """Consume at most max_tokens; completed prompt becomes decode-ready."""
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("prefill chunk budget must be positive")
        with self._operations:
            with self._state:
                if self._poisoned:
                    raise SessionError(-7, "batch request loop needs recovery")
                request = self._tickets.get(ticket)
                if not isinstance(request, _PrefillRequest):
                    return ()
            if request.cancel_requested.is_set():
                return (self._retire(request, "cancelled"),)
            chunk = request.prompt_ids[request.offset:request.offset + max_tokens]
            try:
                if self.prefill_phase_staged:
                    total_consumed = self.batch.append_prefill(
                        request.slot, ticket, chunk, prefill_phase=True)
                else:
                    total_consumed = self.batch.append_prefill(request.slot, ticket, chunk)
                if total_consumed != request.offset + len(chunk):
                    raise SessionError(-7, "native staged prefill reported unexpected progress")
                request.offset = total_consumed
                if request.cancel_requested.is_set():
                    return (self._retire(request, "cancelled"),)
                if request.offset == len(request.prompt_ids):
                    ready = _Request(request.slot, ticket,
                                     self.batch.staged_logits(request.slot),
                                     request.max_tokens, request.stop_ids,
                                     request.temperature, request.top_p, request.rng,
                                     cancel_requested=request.cancel_requested)
                    with self._state:
                        self._prefills.pop(request.slot)
                        self._slots[request.slot] = ready
                        self._tickets[ticket] = ready
                return ()
            except SessionError as exc:
                if exc.status == -9:
                    try:
                        return (self._retire(request, "cancelled"),)
                    except SessionError:
                        pass
                return self._fail_all_after_native_error()

    def request_cancel(self, ticket: int) -> bool:
        """May run concurrently with native decode; stale tickets are inert."""
        with self._state:
            request = self._tickets.get(ticket)
            if request is None:
                return False
            request.cancel_requested.set()
        self.batch.request_cancel(ticket)
        return True

    def _retire(self, request: _Request | _PrefillRequest, reason: str) -> CompletionEvent:
        self.batch.reset_slot(request.slot)
        with self._state:
            self._slots.pop(request.slot, None)
            self._prefills.pop(request.slot, None)
            self._tickets.pop(request.ticket, None)
        return CompletionEvent(request.ticket, request.slot, reason,
                               request.generated if isinstance(request, _Request) else 0)

    def _fail_all_after_native_error(self) -> tuple[CompletionEvent, ...]:
        with self._state:
            active = tuple(self._tickets.values())
            self._poisoned = True
        recovered = True
        for slot in (0, 1):
            try:
                self.batch.reset_slot(slot)
            except SessionError:
                recovered = False
        with self._state:
            self._slots.clear()
            self._prefills.clear()
            self._tickets.clear()
            self._poisoned = not recovered
        return tuple(CompletionEvent(r.ticket, r.slot, "runtime_error",
                                     r.generated if isinstance(r, _Request) else 0)
                     for r in active)

    def advance(self, *, mix_staged: bool = False) -> BatchTick:
        """Sample ready rows; optionally pair one decode with one prompt row."""
        with self._operations:
            with self._state:
                if self._poisoned:
                    raise SessionError(-7, "batch request loop needs recovery")
                active = tuple(self._slots[i] for i in sorted(self._slots))
            emitted: list[TokenEvent] = []
            completed: list[CompletionEvent] = []
            decode_inputs = [0, 0]
            pending_mask = 0
            mixed_ticket = None
            mixed_consumed = None
            try:
                for request in active:
                    if request.cancel_requested.is_set():
                        completed.append(self._retire(request, "cancelled"))
                        continue
                    # The sampler ABI takes float32; a double just below one
                    # can round to 1.0 and violate its [0, 1) contract.
                    draw = min(request.rng.random(), 1.0 - 2.0 ** -24) if request.temperature > 0 else 0.0
                    token = int(self._sample(request.logits, self.batch.vocab_size,
                                             request.temperature, request.top_p,
                                             draw))
                    if token < 0 or token >= self.batch.vocab_size:
                        raise SessionError(-7, "native sampler returned an invalid token")
                    if token in request.stop_ids:
                        completed.append(self._retire(request, "eos"))
                        continue
                    request.generated += 1
                    emitted.append(TokenEvent(request.ticket, request.slot, token))
                    if request.generated >= request.max_tokens:
                        completed.append(self._retire(request, "token_limit"))
                    else:
                        decode_inputs[request.slot] = token
                        pending_mask |= 1 << request.slot
                advanced = 0
                if pending_mask:
                    staged = None
                    if mix_staged and pending_mask in (1, 2):
                        with self._state:
                            staged = self._prefills.get(1 if pending_mask == 1 else 0)
                    if staged is not None and staged.cancel_requested.is_set():
                        completed.append(self._retire(staged, "cancelled"))
                        staged = None
                    if staged is not None:
                        decode_slot = 1 if staged.slot == 0 else 0
                        ready = self._slots[decode_slot]
                        decode_inputs[staged.slot] = staged.prompt_ids[staged.offset]
                        tickets = [0, 0]
                        tickets[decode_slot] = ready.ticket
                        tickets[staged.slot] = staged.ticket
                        try:
                            if self.prefill_phase_staged:
                                advanced, consumed, _ = self.batch.mixed_step(
                                    tuple(decode_inputs), tuple(tickets), staged.slot,
                                    prefill_phase=True)
                            else:
                                advanced, consumed, _ = self.batch.mixed_step(
                                    tuple(decode_inputs), tuple(tickets), staged.slot)
                        except SessionError as exc:
                            if exc.status != -9 or not (ready.cancel_requested.is_set() or
                                                         staged.cancel_requested.is_set()):
                                raise
                            # Native rejection is before execution. Retire
                            # cancelled owners, then advance an uncancelled
                            # decode row through the ordinary one-row path.
                            if staged.cancel_requested.is_set():
                                completed.append(self._retire(staged, "cancelled"))
                            if ready.cancel_requested.is_set():
                                completed.append(self._retire(ready, "cancelled"))
                                pending_mask = 0
                                advanced = 0
                            else:
                                advanced, _ = self.batch.step(tuple(decode_inputs))
                        else:
                            if advanced != 3 or consumed != staged.offset + 1:
                                raise SessionError(-7, "native mixed step reported invalid progress")
                            staged.offset = consumed
                            mixed_ticket = staged.ticket
                            mixed_consumed = consumed
                            if staged.cancel_requested.is_set():
                                completed.append(self._retire(staged, "cancelled"))
                            elif staged.offset == len(staged.prompt_ids):
                                ready_prefill = _Request(
                                    staged.slot, staged.ticket,
                                    self.batch.staged_logits(staged.slot),
                                    staged.max_tokens, staged.stop_ids,
                                    staged.temperature, staged.top_p, staged.rng,
                                    cancel_requested=staged.cancel_requested)
                                with self._state:
                                    self._prefills.pop(staged.slot)
                                    self._slots[staged.slot] = ready_prefill
                                    self._tickets[staged.ticket] = ready_prefill
                    else:
                        advanced, _ = self.batch.step(tuple(decode_inputs))
                    if mixed_ticket is None:
                        if advanced & ~pending_mask:
                            raise SessionError(-7, "native step advanced an unscheduled slot")
                        if advanced != pending_mask:
                            with self._state:
                                missing = [self._slots[i] for i in (0, 1)
                                           if (pending_mask & (1 << i)) and not (advanced & (1 << i))]
                            if any(not r.cancel_requested.is_set() for r in missing):
                                raise SessionError(-7, "native step omitted an active slot")
                            for request in missing:
                                completed.append(self._retire(request, "cancelled"))
                with self._state:
                    newly_cancelled = tuple(r for r in self._slots.values()
                                            if r.cancel_requested.is_set())
                for request in newly_cancelled:
                    completed.append(self._retire(request, "cancelled"))
                return BatchTick(tuple(emitted), tuple(completed), advanced,
                                 mixed_ticket, mixed_consumed)
            except SessionError:
                completed.extend(self._fail_all_after_native_error())
                # A failing native step may have advanced neither, one, or
                # both slots. The ABI explicitly leaves advancement unknown.
                return BatchTick(tuple(emitted), tuple(completed), None)

    def active_tickets(self) -> tuple[int, ...]:
        with self._state:
            return tuple(r.ticket for _, r in sorted(
                ((r.slot, r) for r in self._tickets.values())))

    def recover(self) -> None:
        """Retry both native resets after a failed cleanup; never reuse early."""
        with self._operations:
            with self._state:
                if self._tickets:
                    raise SessionError(-6, "cannot recover while requests are active")
            failed = False
            for slot in (0, 1):
                try:
                    self.batch.reset_slot(slot)
                except SessionError:
                    failed = True
            with self._state:
                self._poisoned = failed
            if failed:
                raise SessionError(-7, "batch request loop recovery did not reset both slots")
