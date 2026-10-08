#ifndef CK_SESSION_BATCH_TICKET_V8_H
#define CK_SESSION_BATCH_TICKET_V8_H

#include <stdatomic.h>
#include <stdint.h>

/* The low bit records cancellation; upper bits identify a unique admission. */
static inline uint_fast64_t ck_batch_ticket_state(uint64_t ticket) {
    return (uint_fast64_t)(ticket << 1);
}

static inline int ck_batch_ticket_try_cancel_observed(
    atomic_uint_fast64_t *state, uint_fast64_t observed, uint64_t ticket) {
    if (!ticket || (observed >> 1) != ticket || (observed & 1u))
        return 0;
    return atomic_compare_exchange_strong_explicit(
        state, &observed, observed | 1u,
        memory_order_acq_rel, memory_order_acquire);
}

#endif
