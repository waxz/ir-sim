/*
 * shm_types.h -- shared memory layout for the IR-SIM ↔ C++ robot bridge.
 *
 * Both the Python bridge (irsim/util/shm_bridge.py) and the C++ controller
 * map the same POSIX shm segment.  The layout is fully specified here; keep
 * shm_bridge.py's ctypes structs in sync with any changes made here.
 *
 * Shared block map (384 bytes total):
 *
 *   offset   0 :  IrsimHeader   (128 B)  -- ready flag
 *   offset 128 :  IrsimStateSlot (128 B) -- robot state, written by IR-SIM
 *   offset 256 :  IrsimCmdSlot   (128 B) -- velocity cmd, written by C++
 *
 * Seqlock protocol (single writer, multiple readers, wait-free reads):
 *
 *   writer:
 *     slot->seq++            -- seq becomes odd (writing)
 *     fence()
 *     write data
 *     fence()
 *     slot->seq2 = slot->seq -- seq2 == seq, even (valid)
 *
 *   reader:
 *     do {
 *       s1   = slot->seq;
 *       fence();
 *       read data;
 *       fence();
 *       s2   = slot->seq2;
 *     } while (s1 != s2 || s1 & 1);
 *
 * Memory ordering: on x86-64 (TSO) __sync_synchronize() is sufficient.
 * On AArch64, replace with:
 *   __asm__ volatile("dmb ish" ::: "memory")
 */

#pragma once
#include <stdint.h>
#include <string.h>

#define IRSIM_SHM_NAME  "/irsim_bridge_v1"
#define IRSIM_SHM_SIZE  512

/* ── Robot state (72 bytes) ─────────────────────────────────────────────── */
/* Written by IR-SIM after every env.step(). Read by the C++ controller.    */
typedef struct {
    double   x, y, heading;          /* world-frame pose (m, m, rad)        */
    float    vx, vy, omega;          /* world-frame velocity (m/s, rad/s)   */
    float    goal_x, goal_y;         /* current goal position (m)           */
    float    goal_dist;              /* Euclidean distance to goal (m)       */
    uint64_t step;                   /* sim step counter (increases by 1)   */
    double   sim_time;               /* simulated time (s)                  */
    uint8_t  reached;                /* 1 when robot reached the goal       */
    uint8_t  collision;              /* 1 when robot is in collision         */
    uint8_t  _pad[6];
} IrsimState; /* sizeof == 72 */

/* ── Velocity command (16 bytes) ────────────────────────────────────────── */
/* Written by C++ controller. Read by IR-SIM before every env.step().       */
typedef struct {
    float    linear;                 /* forward velocity (m/s)              */
    float    angular;                /* angular velocity (rad/s, CCW+)      */
    uint32_t seq;                    /* command counter (monotone increment) */
    uint32_t valid;                  /* nonzero = command is fresh          */
} IrsimCmd; /* sizeof == 16 */

/* ── Seqlock wrappers (128 bytes each, one cache-line aligned) ─────────── */
typedef struct {
    volatile uint64_t seq;           /* odd while writing                   */
    IrsimState        state;         /* 72 bytes                            */
    volatile uint64_t seq2;          /* mirrors seq when valid              */
    uint8_t           _fill[40];     /* pad to 128                          */
} IrsimStateSlot; /* sizeof == 128 */

typedef struct {
    volatile uint64_t seq;           /* odd while writing                   */
    IrsimCmd          cmd;           /* 16 bytes                            */
    volatile uint64_t seq2;          /* mirrors seq when valid              */
    uint8_t           _fill[96];     /* pad to 128                          */
} IrsimCmdSlot; /* sizeof == 128 */

typedef struct {
    volatile uint64_t ready;         /* 1 = segment is initialized          */
    uint8_t           _fill[120];    /* pad to 128                          */
} IrsimHeader; /* sizeof == 128 */

/* ── Full shared block ──────────────────────────────────────────────────── */
typedef struct {
    IrsimHeader    header;           /* offset   0                          */
    IrsimStateSlot state;            /* offset 128                          */
    IrsimCmdSlot   cmd;              /* offset 256                          */
} IrsimBlock; /* sizeof == 384 */

/* ── Inline helpers ─────────────────────────────────────────────────────── */

static inline void irsim_fence(void) {
    __sync_synchronize();
}

static inline void irsim_write_state(IrsimStateSlot *slot,
                                      const IrsimState *s) {
    slot->seq++;
    irsim_fence();
    memcpy((void *)&slot->state, s, sizeof(*s));
    irsim_fence();
    slot->seq2 = slot->seq;
}

/* Returns 0 on a clean read, -1 if the slot is mid-write (caller should
 * retry or use the previous value). */
static inline int irsim_read_state(const IrsimStateSlot *slot,
                                    IrsimState *out) {
    uint64_t s1, s2;
    s1 = slot->seq;
    irsim_fence();
    memcpy(out, (const void *)&slot->state, sizeof(*out));
    irsim_fence();
    s2 = slot->seq2;
    return (s1 == s2 && !(s1 & 1)) ? 0 : -1;
}

static inline void irsim_write_cmd(IrsimCmdSlot *slot,
                                    const IrsimCmd *c) {
    slot->seq++;
    irsim_fence();
    memcpy((void *)&slot->cmd, c, sizeof(*c));
    irsim_fence();
    slot->seq2 = slot->seq;
}

static inline int irsim_read_cmd(const IrsimCmdSlot *slot, IrsimCmd *out) {
    uint64_t s1, s2;
    s1 = slot->seq;
    irsim_fence();
    memcpy(out, (const void *)&slot->cmd, sizeof(*out));
    irsim_fence();
    s2 = slot->seq2;
    return (s1 == s2 && !(s1 & 1)) ? 0 : -1;
}
