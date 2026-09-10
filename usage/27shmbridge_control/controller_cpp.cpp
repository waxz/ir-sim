/**
 * controller_cpp.cpp — C++ proportional-heading controller (shmbridge subscriber).
 *
 * Attaches to the shared-memory segment published by sim.py (Python IR-SIM),
 * reads robot state, and writes back velocity commands at 40 Hz.
 *
 * Build (from repo root):
 *   g++ -std=c++17 -O2 \
 *       -I shmbridge/include \
 *       usage/27shmbridge_control/controller_cpp.cpp \
 *       -lrt -lpthread \
 *       -o /tmp/ctrl_cpp
 *
 * Run (while sim.py is running in another terminal):
 *   /tmp/ctrl_cpp
 */

#include <shmbridge/core.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <thread>

using namespace shmbridge;
using clk = std::chrono::steady_clock;

// ── tunables ──────────────────────────────────────────────────────────────
static constexpr const char* SHM_NAME  = "/irsim_shmbridge_demo";
static constexpr double      RATE_HZ   = 40.0;
static constexpr float       V_MAX     = 1.0f;    // m/s
static constexpr float       OMEGA_MAX = 1.5f;    // rad/s
static constexpr float       K_ANG     = 2.0f;    // P-gain for heading error
static constexpr float       GOAL_TOL  = 0.3f;    // m — stop threshold
// ─────────────────────────────────────────────────────────────────────────

static std::atomic<bool> g_stop{false};

static void on_signal(int) { g_stop = true; }

/** Signed angle from current heading to the goal direction, wrapped to [-π, π]. */
static float bearing_error(const RobotState& s) {
    float dx  = static_cast<float>(s.goal_x - s.x);
    float dy  = static_cast<float>(s.goal_y - s.y);
    float tgt = std::atan2(dy, dx);
    float err = tgt - static_cast<float>(s.heading);
    // Wrap
    while (err >  static_cast<float>(M_PI)) err -= 2.0f * static_cast<float>(M_PI);
    while (err < -static_cast<float>(M_PI)) err += 2.0f * static_cast<float>(M_PI);
    return err;
}

int main() {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    ShmSubscriber sub(SHM_NAME);
    std::printf("[ctrl-cpp] attaching to %s …\n", SHM_NAME);
    try {
        sub.attach(10'000.0);  // wait up to 10 s for publisher
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ctrl-cpp] attach failed: %s\n", e.what());
        return 1;
    }
    std::printf("[ctrl-cpp] attached\n");

    const auto period = std::chrono::duration<double>(1.0 / RATE_HZ);
    auto next = clk::now();

    uint32_t cmd_seq  = 0;
    uint64_t last_sim_step = static_cast<uint64_t>(-1);

    while (!g_stop) {
        // Spin-read with 100 ms deadline
        auto state_opt = sub.read_state_spin(0, 200.0);
        if (!state_opt) {
            std::printf("[ctrl-cpp] timeout — sim may have exited\n");
            break;
        }

        const RobotState& s = *state_opt;

        if (s.reached || s.collision) {
            const char* why = s.reached ? "reached" : "collision";
            std::printf("[ctrl-cpp] %s  x=%.2f  y=%.2f  step=%llu\n",
                        why, s.x, s.y,
                        static_cast<unsigned long long>(s.step));
            break;
        }

        // Only compute a new command when the sim published a new step.
        if (s.step != last_sim_step) {
            last_sim_step = s.step;

            float linear = 0.0f, angular = 0.0f;

            float dist = static_cast<float>(s.goal_dist);
            if (dist >= GOAL_TOL) {
                float err = bearing_error(s);
                angular = std::max(-OMEGA_MAX, std::min(OMEGA_MAX, K_ANG * err));
                linear  = V_MAX * std::max(0.0f, std::cos(err));
            }

            sub.write_cmd(0, 0, linear, angular, cmd_seq++);

            if (s.step % 20 == 0) {
                std::printf("[ctrl-cpp] step=%4llu  x=%.2f  y=%.2f"
                            "  h=%.1f°  dist=%.2f  cmd=(%.2f,%.2f)\n",
                            static_cast<unsigned long long>(s.step),
                            s.x, s.y,
                            static_cast<float>(s.heading) * 180.0f /
                                static_cast<float>(M_PI),
                            dist, linear, angular);
            }
        }

        next += period;
        std::this_thread::sleep_until(next);
    }

    std::printf("[ctrl-cpp] done\n");
    return 0;
}
