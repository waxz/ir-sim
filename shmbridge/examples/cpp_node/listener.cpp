/*
 * listener.cpp — receives Pose2d messages from talker using the MessageQueue API.
 *
 * Demonstrates create_queue<T>(): spin_once() pushes messages into the queue;
 * the control loop drains at its own rate — decoupling transport timing from
 * processing timing.
 *
 * Run (start talker first):
 *   ./build/listener
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
    auto node = sb::make_node("listener");

    /* create_queue returns {subscription_handle, shared_ptr<MessageQueue<T>>}.
     * SensorDataQoS (depth=1) → seqlock transport, keep-latest. */
    auto [sub, queue] = node->create_queue<msg::Pose2d>("robot/pose",
                                                         sb::SensorDataQoS());
    (void)sub; /* handle kept alive by node; unused directly */

    std::printf("listener: waiting for robot/pose (publisher may start later)\n");

    uint64_t count = 0;
    while (sb::g_ok.load()) {
        /* spin_once: non-blocking; attaches lazily when publisher appears */
        node->spin_once();

        /* Drain queue: pop_latest() keeps only the freshest message */
        if (auto pose = queue->pop_latest()) {
            uint64_t age_us = (shmbridge::detail::now_ns() - pose->stamp_ns) / 1000;
            if (count % 100 == 0)
                std::printf("listener: x=%.2f m  age=%llu µs\n",
                            pose->x, (unsigned long long)age_us);
            ++count;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    sb::shutdown();
    std::printf("listener: received %llu messages\n", (unsigned long long)count);
}
