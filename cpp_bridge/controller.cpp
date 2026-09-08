/*
 * controller.cpp -- minimal proportional heading controller for IR-SIM.
 *
 * Maps the POSIX shm segment created by the Python bridge, polls robot
 * state, computes (linear, angular) velocity commands, and writes them
 * back.  Runs as a separate OS process alongside the IR-SIM step loop.
 *
 * Build (from ir-sim/cpp_bridge/):
 *   cmake -B build && cmake --build build
 *
 * Run (start IR-SIM demo first, then this):
 *   ./build/controller
 */

#include "shm_types.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static volatile bool g_running = true;
static void on_signal(int) { g_running = false; }

/* ── shm attach ─────────────────────────────────────────────────────────── */

static IrsimBlock *shm_attach(int timeout_sec = 30) {
    fprintf(stderr, "controller: waiting for segment '%s' ...\n",
            IRSIM_SHM_NAME);

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(timeout_sec);
    int fd = -1;
    while (std::chrono::steady_clock::now() < deadline) {
        fd = shm_open(IRSIM_SHM_NAME, O_RDWR, 0);
        if (fd >= 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (fd < 0) {
        fprintf(stderr, "controller: shm_open timed out: %s\n",
                strerror(errno));
        return nullptr;
    }

    void *p = mmap(nullptr, IRSIM_SHM_SIZE,
                   PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) {
        fprintf(stderr, "controller: mmap failed: %s\n", strerror(errno));
        return nullptr;
    }

    IrsimBlock *blk = reinterpret_cast<IrsimBlock *>(p);
    while (blk->header.ready == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    fprintf(stderr, "controller: attached\n");
    return blk;
}

/* ── proportional heading controller ────────────────────────────────────── */
/*
 * Strategy: point nose at goal (angular P controller), drive forward when
 * heading error is small.  Simple but enough to validate the bridge.
 */
static void compute_cmd(const IrsimState &s, IrsimCmd &cmd,
                        uint32_t &cmd_seq) {
    if (s.reached || s.collision || s.goal_dist < 0.10f) {
        cmd = {0.0f, 0.0f, ++cmd_seq, 1};
        return;
    }

    const float dx  = s.goal_x - (float)s.x;
    const float dy  = s.goal_y - (float)s.y;
    const float tgt = std::atan2(dy, dx);

    float err = tgt - (float)s.heading;
    while (err >  (float)M_PI) err -= 2.0f * (float)M_PI;
    while (err < -(float)M_PI) err += 2.0f * (float)M_PI;

    const float K_ang    = 1.5f;
    const float K_lin    = 0.5f;
    const float ang_dead = 0.3f;   /* radians -- drive forward inside this */

    float angular = K_ang * err;
    if (angular >  1.5f) angular =  1.5f;
    if (angular < -1.5f) angular = -1.5f;

    float linear = (std::fabs(err) < ang_dead) ? K_lin : 0.0f;
    /* slow down near goal */
    if (s.goal_dist < 0.5f) linear *= (s.goal_dist / 0.5f);

    cmd = {linear, angular, ++cmd_seq, 1};
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(void) {
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    IrsimBlock *blk = shm_attach();
    if (!blk) return 1;

    IrsimState state{};
    IrsimCmd   cmd{};
    uint32_t   cmd_seq  = 0;
    uint64_t   last_step = UINT64_MAX;
    uint64_t   iters     = 0;

    fprintf(stderr, "controller: running -- Ctrl+C to stop\n\n");
    auto t0 = std::chrono::steady_clock::now();

    while (g_running) {
        /* non-blocking seqlock read */
        if (irsim_read_state(&blk->state, &state) != 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }

        /* skip if no new step has been published yet */
        if (state.step == last_step) {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
            continue;
        }
        last_step = state.step;
        ++iters;

        compute_cmd(state, cmd, cmd_seq);
        irsim_write_cmd(&blk->cmd, &cmd);

        if (iters % 50 == 0) {
            const double dt = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr,
                    "step=%5llu  pos=(%.2f, %.2f)  h=%.2f  "
                    "dist=%.2f  cmd=(%.2f, %.2f)  %.0f Hz\n",
                    (unsigned long long)state.step,
                    state.x, state.y, state.heading,
                    state.goal_dist,
                    cmd.linear, cmd.angular,
                    iters / dt);
        }
    }

    munmap(blk, IRSIM_SHM_SIZE);
    fprintf(stderr, "\ncontroller: clean exit\n");
    return 0;
}
