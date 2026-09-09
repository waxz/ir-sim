/*
 * shmbridge/shmbridge.h  –  header-only C/C++ library for the shmbridge protocol.
 *
 * Usage (C++):
 *
 *   #include <shmbridge/shmbridge.h>
 *
 *   IrsimBlock *blk = shmbridge_attach(SHMBRIDGE_SHM_NAME);
 *   if (!blk) { perror("shmbridge_attach"); return 1; }
 *
 *   IrsimState s; while (irsim_read_state(&blk->states[0], &s) != 0) {}
 *   IrsimCmd   c = {0.5f, 0.0f, 1, 1};
 *   irsim_write_cmd(&blk->cmds[0], &c);
 *
 * Memory ordering
 * ───────────────
 * The fence() call uses:
 *   • C11/C++11 atomic_thread_fence(memory_order_seq_cst) when available
 *   • __asm__ volatile("dmb ish" ::: "memory") on AArch64 (armv8+) otherwise
 *   • __sync_synchronize() as a last-resort fallback
 *
 * On x86-64 (TSO) all three reduce to a no-op plus a compiler barrier,
 * which is all that is needed.  On AArch64 (Jetson, Pi 5, M-series Mac)
 * the explicit barrier is required to prevent load/store reordering around
 * the seqlock counters.
 */

#pragma once

#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#ifdef __cplusplus
#  include <atomic>
#  define _SB_FENCE() std::atomic_thread_fence(std::memory_order_seq_cst)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__STDC_NO_ATOMICS__)
#  include <stdatomic.h>
#  define _SB_FENCE() atomic_thread_fence(memory_order_seq_cst)
#elif defined(__aarch64__) || defined(__ARM_ARCH_8A__)
#  define _SB_FENCE() __asm__ volatile("dmb ish" ::: "memory")
#else
#  define _SB_FENCE() __sync_synchronize()
#endif

/* ── Monotonic clock helper ─────────────────────────────────────────────── */

static inline uint64_t shmbridge_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ── Attach to an existing segment (no unlink on detach) ───────────────── */

/*
 * shmbridge_attach() – open and mmap an existing shmbridge segment.
 *
 * Blocks until header.ready == 1 (up to timeout_ms milliseconds).
 * Returns a pointer into the mmap'd region on success, NULL on failure.
 * The caller must never munmap or shm_unlink the returned pointer; use
 * shmbridge_detach() instead.
 */
static inline IrsimBlock *shmbridge_attach(
    const char *shm_name,
    unsigned    n_robots,
    unsigned    timeout_ms)
{
    int fd = shm_open(shm_name, O_RDWR, 0666);
    if (fd < 0) return NULL;

    size_t sz = (SHMBRIDGE_SHM_SIZE_ALIGNED(n_robots));
    void *ptr = mmap(NULL, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return NULL;

    IrsimBlock *blk = (IrsimBlock *)ptr;

    /* Wait for Python sim to signal ready. */
    uint64_t deadline_ns = shmbridge_now_ns() + (uint64_t)timeout_ms * 1000000ULL;
    while (!blk->header.ready) {
        if (shmbridge_now_ns() > deadline_ns) {
            munmap(ptr, sz);
            return NULL;
        }
        struct timespec ts = {0, 500000};  /* 0.5 ms */
        nanosleep(&ts, NULL);
    }

    /* Schema check (non-fatal: v1 segments have magic == 0, skip check). */
    if (blk->header.magic != 0 && blk->header.magic != SHMBRIDGE_MAGIC) {
        munmap(ptr, sz);
        return NULL;
    }

    return blk;
}

/* shmbridge_attach() with default 30-second timeout, single-robot. */
#define shmbridge_attach1(name) shmbridge_attach((name), 1, 30000)

static inline void shmbridge_detach(IrsimBlock *blk, unsigned n_robots) {
    if (blk) munmap(blk, SHMBRIDGE_SHM_SIZE_ALIGNED(n_robots));
}

/* ── Seqlock read / write helpers ──────────────────────────────────────── */

static inline void irsim_write_state(IrsimStateSlot *slot, const IrsimState *s) {
    slot->seq++;           /* even → odd  (begin write) */
    _SB_FENCE();
    memcpy((void *)&slot->state, s, sizeof(*s));
    _SB_FENCE();
    slot->seq++;           /* odd  → even (write done)  */
    slot->seq2         = slot->seq;
    slot->writer_ts_ns = shmbridge_now_ns();  /* heartbeat */
}

/* Returns 0 on a clean read, -1 if slot is mid-write (caller should retry). */
static inline int irsim_read_state(const IrsimStateSlot *slot, IrsimState *out) {
    uint64_t s1, s2;
    s1 = slot->seq;
    _SB_FENCE();
    memcpy(out, (const void *)&slot->state, sizeof(*out));
    _SB_FENCE();
    s2 = slot->seq2;
    return (s1 == s2 && !(s1 & 1)) ? 0 : -1;
}

static inline void irsim_write_cmd(IrsimCmdSlot *slot, const IrsimCmd *c) {
    slot->seq++;
    _SB_FENCE();
    memcpy((void *)&slot->cmd, c, sizeof(*c));
    _SB_FENCE();
    slot->seq++;
    slot->seq2         = slot->seq;
    slot->writer_ts_ns = shmbridge_now_ns();  /* heartbeat */
}

static inline int irsim_read_cmd(const IrsimCmdSlot *slot, IrsimCmd *out) {
    uint64_t s1, s2;
    s1 = slot->seq;
    _SB_FENCE();
    memcpy(out, (const void *)&slot->cmd, sizeof(*out));
    _SB_FENCE();
    s2 = slot->seq2;
    return (s1 == s2 && !(s1 & 1)) ? 0 : -1;
}

/* ── Liveness checks ───────────────────────────────────────────────────── */

/*
 * Returns 1 if the last write to *slot* was within max_age_ms milliseconds,
 * 0 if it is stale, or 1 if no write has ever occurred (writer_ts_ns == 0).
 */
static inline int irsim_state_alive(const IrsimStateSlot *slot, uint64_t max_age_ms) {
    uint64_t ts = slot->writer_ts_ns;
    if (ts == 0) return 1;
    return (shmbridge_now_ns() - ts) < max_age_ms * 1000000ULL;
}

static inline int irsim_cmd_alive(const IrsimCmdSlot *slot, uint64_t max_age_ms) {
    uint64_t ts = slot->writer_ts_ns;
    if (ts == 0) return 1;
    return (shmbridge_now_ns() - ts) < max_age_ms * 1000000ULL;
}

#ifdef __cplusplus
/*
 * C++ RAII wrapper.  Usage:
 *
 *   ShmBridgeClient client("/irsim_bridge_v2");
 *   IrsimState s; while (client.read_state(s) != 0) {}
 */
struct ShmBridgeClient {
    IrsimBlock *blk    = nullptr;
    unsigned    n      = 1;

    explicit ShmBridgeClient(const char *name, unsigned n_robots = 1,
                              unsigned timeout_ms = 30000)
        : n(n_robots)
    {
        blk = shmbridge_attach(name, n_robots, timeout_ms);
    }

    ~ShmBridgeClient() { shmbridge_detach(blk, n); blk = nullptr; }

    bool connected() const { return blk != nullptr; }

    int  read_state(IrsimState &out, unsigned idx = 0) const {
        return irsim_read_state(&blk->states[idx], &out);
    }
    void write_cmd(const IrsimCmd &c, unsigned idx = 0) {
        irsim_write_cmd(&blk->cmds[idx], &c);
    }
    bool state_alive(uint64_t max_age_ms = 100, unsigned idx = 0) const {
        return irsim_state_alive(&blk->states[idx], max_age_ms);
    }

    ShmBridgeClient(const ShmBridgeClient &) = delete;
    ShmBridgeClient &operator=(const ShmBridgeClient &) = delete;
};
#endif /* __cplusplus */
