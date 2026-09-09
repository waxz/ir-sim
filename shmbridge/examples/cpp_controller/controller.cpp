/*
 * controller.cpp – example C++ heading controller using shmbridge v2.
 *
 * Build:
 *   cmake -B build && cmake --build build
 *
 * Run (after starting the Python sim):
 *   ./build/controller [shm_name]   (default: /irsim_bridge_v2)
 *
 * Features:
 *   – Proportional heading + speed controller
 *   – Liveness check: warns if sim goes stale (> 200 ms)
 *   – AArch64-safe seqlock barriers via shmbridge.h
 *   – Schema version check on attach
 */

#include <shmbridge/shmbridge.h>
#include <cmath>
#include <cstdio>
#include <cstring>

static constexpr float KP_HEADING   = 1.5f;   // rad/s per radian error
static constexpr float KP_SPEED     = 0.5f;   // m/s per metre to goal
static constexpr float MAX_SPEED    = 1.0f;   // m/s cap
static constexpr float GOAL_RADIUS  = 0.5f;   // stop threshold (m)
static constexpr float HEADING_DEAD = 0.3f;   // rad dead-zone for fwd motion

int main(int argc, char **argv) {
    const char *shm_name = (argc > 1) ? argv[1] : SHMBRIDGE_SHM_NAME;
    printf("Connecting to %s ...\n", shm_name);

    ShmBridgeClient client(shm_name, 1, 30000);
    if (!client.connected()) {
        fprintf(stderr, "shmbridge_attach failed – is the sim running?\n");
        return 1;
    }

    printf("Connected. Schema v%u, %u robot(s).\n",
           client.blk->header.schema_version,
           (unsigned)client.blk->header.n_robots);

    uint64_t last_step = UINT64_MAX;
    uint32_t cmd_seq   = 0;

    while (true) {
        // ── 1. Read state (retry on torn read) ──────────────────────────
        IrsimState s;
        while (client.read_state(s) != 0) {}

        if (s.step == last_step) {
            struct timespec ts = {0, 100000}; /* 0.1 ms */
            nanosleep(&ts, NULL);
            continue;
        }
        last_step = s.step;

        // ── 2. Liveness: warn if sim writes are stale ────────────────────
        if (!client.state_alive(200)) {
            fprintf(stderr, "WARN: sim state stale (> 200 ms)\n");
        }

        // ── 3. Check goal reached ────────────────────────────────────────
        if (s.reached || s.goal_dist < GOAL_RADIUS) {
            IrsimCmd stop = {0.0f, 0.0f, ++cmd_seq, 1};
            client.write_cmd(stop);
            printf("step %llu: goal reached, stopped.\n",
                   (unsigned long long)s.step);
            break;
        }

        // ── 4. Proportional controller ───────────────────────────────────
        float goal_angle = std::atan2(s.goal_y - s.y, s.goal_x - s.x);
        float err        = goal_angle - s.heading;
        // wrap to (-π, π)
        while (err >  (float)M_PI) err -= 2.0f * (float)M_PI;
        while (err < -(float)M_PI) err += 2.0f * (float)M_PI;

        float angular = KP_HEADING * err;
        float linear  = 0.0f;
        if (std::fabs(err) < HEADING_DEAD) {
            float raw = KP_SPEED * s.goal_dist;
            linear = raw < MAX_SPEED ? raw : MAX_SPEED;
        }

        IrsimCmd cmd = {linear, angular, ++cmd_seq, 1};
        client.write_cmd(cmd);

        if (s.step % 100 == 0) {
            printf("step %6llu  pos(%.2f,%.2f)  head%.2f  goal_dist %.2f  "
                   "cmd(%.2f,%.2f)\n",
                   (unsigned long long)s.step,
                   s.x, s.y, s.heading, s.goal_dist, linear, angular);
        }
    }
    return 0;
}
