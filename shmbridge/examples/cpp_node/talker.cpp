/*
 * talker.cpp — publishes Pose2d at 100 Hz using the shmbridge Node API.
 *
 * Build:
 *   cmake -B build -DSHMBRIDGE_BUILD_EXAMPLES=ON && cmake --build build
 *
 * Run (start talker first, then listener in a second terminal):
 *   ./build/talker
 */

#include <shmbridge/node.hpp>
#include <shmbridge/messages.hpp>

#include <csignal>
#include <cstdio>
#include <chrono>
#include <thread>

namespace sb  = shmbridge::ros_compat;
namespace msg = shmbridge::msg;

int main() {
    std::signal(SIGINT,  [](int) { sb::g_ok = false; });
    std::signal(SIGTERM, [](int) { sb::g_ok = false; });

    sb::init();
    auto node = sb::make_node("talker");

    /* SensorDataQoS: depth=1, keep-latest → seqlock transport */
    auto pub = node->create_publisher<msg::Pose2d>("robot/pose", sb::SensorDataQoS());

    std::printf("talker: publishing robot/pose at 100 Hz\n");

    msg::Pose2d pose;
    unsigned step = 0;

    while (sb::g_ok.load()) {
        pose.x        = step * 0.01;   /* move 1 cm per step */
        pose.y        = 0.0;
        pose.heading  = 0.0;
        pose.stamp_ns = shmbridge::detail::now_ns();

        pub->publish(pose);            /* zero-overhead direct handle path */

        if (step % 100 == 0)
            std::printf("talker: step=%u  x=%.2f m\n", step, pose.x);

        ++step;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    sb::shutdown();
    std::printf("talker: stopped\n");
}
