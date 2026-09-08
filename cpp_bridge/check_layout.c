/*
 * check_layout.c -- verify that struct sizes and offsets match the Python
 * ctypes declarations in irsim/util/shm_bridge.py.  Build and run once.
 *
 *   ./build/check_layout
 *
 * All lines must print PASS.
 */
#include "shm_types.h"
#include <stdio.h>
#include <stddef.h>
#include <assert.h>

#define CHECK_SIZE(T, expected)                                          \
    do {                                                                 \
        int ok = (sizeof(T) == (expected));                              \
        printf("%s  sizeof(" #T ") = %zu (expected %d)\n",              \
               ok ? "PASS" : "FAIL", sizeof(T), (expected));            \
    } while (0)

#define CHECK_OFFSET(T, field, expected)                                 \
    do {                                                                 \
        int ok = (offsetof(T, field) == (expected));                     \
        printf("%s  offsetof(" #T ", " #field ") = %zu (expected %d)\n",\
               ok ? "PASS" : "FAIL", offsetof(T, field), (expected));   \
    } while (0)

int main(void) {
    puts("=== struct size checks ===");
    CHECK_SIZE(IrsimState,     72);
    CHECK_SIZE(IrsimCmd,       16);
    CHECK_SIZE(IrsimStateSlot, 128);
    CHECK_SIZE(IrsimCmdSlot,   128);
    CHECK_SIZE(IrsimHeader,    128);
    CHECK_SIZE(IrsimBlock,     384);

    puts("\n=== IrsimState field offsets ===");
    CHECK_OFFSET(IrsimState, x,         0);
    CHECK_OFFSET(IrsimState, y,         8);
    CHECK_OFFSET(IrsimState, heading,   16);
    CHECK_OFFSET(IrsimState, vx,        24);
    CHECK_OFFSET(IrsimState, vy,        28);
    CHECK_OFFSET(IrsimState, omega,     32);
    CHECK_OFFSET(IrsimState, goal_x,    36);
    CHECK_OFFSET(IrsimState, goal_y,    40);
    CHECK_OFFSET(IrsimState, goal_dist, 44);
    CHECK_OFFSET(IrsimState, step,      48);
    CHECK_OFFSET(IrsimState, sim_time,  56);
    CHECK_OFFSET(IrsimState, reached,   64);
    CHECK_OFFSET(IrsimState, collision, 65);

    puts("\n=== IrsimBlock slot offsets ===");
    CHECK_OFFSET(IrsimBlock, header, 0);
    CHECK_OFFSET(IrsimBlock, state,  128);
    CHECK_OFFSET(IrsimBlock, cmd,    256);

    return 0;
}
