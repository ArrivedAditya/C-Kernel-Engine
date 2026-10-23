"""Plan external generated-prefill calls under a selected row-group contract.

The generated model exports the alignment derived from its selected prefill
providers. A nonfinal call must end on that alignment: a tail reduction cannot
be retroactively replaced by a full-group reduction on the next call. Rows
from different requests are never combined by this planner. Mixed-request
packing needs a separate provider contract and numerical certification.
"""

from __future__ import annotations


def plan_prefill_chunks(total_tokens: int, max_chunk_tokens: int,
                        row_group: int, *, initial_prefix: int = 0,
                        packing: str = "isolated") -> tuple[tuple[int, int], ...]:
    """Return (absolute start, count) for a complete, bounded prompt.

    ``initial_prefix`` represents a separately submitted first call. Its
    boundary must be aligned before any subsequent append is admitted. The
    final call may contain a partial group because it has no later append.
    """
    if any(type(value) is not int for value in
           (total_tokens, max_chunk_tokens, row_group, initial_prefix)):
        raise ValueError("prefill geometry must use integer values")
    if not 0 < total_tokens <= 0x7FFFFFFF or max_chunk_tokens < 1 or row_group < 1:
        raise ValueError("invalid prefill token count, chunk budget or row group")
    if not 0 <= initial_prefix < total_tokens:
        raise ValueError("initial prefix must precede the final token")
    if packing != "isolated":
        raise ValueError("cross-request row packing has no certified reduction contract")
    if initial_prefix and initial_prefix % row_group:
        raise ValueError("initial prefix ends inside a provider row group")
    calls: list[tuple[int, int]] = []
    position = 0
    if initial_prefix:
        calls.append((0, initial_prefix))
        position = initial_prefix
    while total_tokens - position > max_chunk_tokens:
        count = (max_chunk_tokens // row_group) * row_group
        if count == 0:
            raise ValueError("chunk budget cannot reach the next aligned boundary")
        calls.append((position, count))
        position += count
    calls.append((position, total_tokens - position))
    return tuple(calls)
