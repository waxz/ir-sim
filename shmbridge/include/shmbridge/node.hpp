/*
 * shmbridge/node.hpp — ROS 2-style Node API built on shmbridge.
 *
 * Mirrors the rclcpp surface closely enough that porting is mechanical:
 *
 *   // rclcpp                              // shmbridge::ros_compat
 *   rclcpp::init(argc, argv);              shmbridge::ros_compat::init();
 *   auto node = rclcpp::Node("cam");       auto node = ros_compat::make_node("cam");
 *   auto pub  = node->create_publisher     auto pub  = node->create_publisher
 *                 <Pose2d>("pose", 10);                <Pose2d>("pose", 10);
 *   auto sub  = node->create_subscription  auto sub  = node->create_subscription
 *                 <Pose2d>("pose", 10, cb);            <Pose2d>("pose", 10, cb);
 *   rclcpp::spin(node);                    ros_compat::spin(node);
 *   rclcpp::shutdown();                    ros_compat::shutdown();
 *
 * Differences from rclcpp worth noting:
 *
 *   - Subscription<T> is a stateless handle.  The callback fires from
 *     spin_once() in the same thread — no executor thread pool.
 *   - QoS(depth<=1) maps to a seqlock (keep-latest) topic; depth>1 maps to a
 *     SPSC ring.  Reliable vs BestEffort is currently treated the same.
 *   - create_publisher() opens the seqlock / ring segment immediately.
 *     create_subscription() attaches at spin_once() once a publisher is
 *     found via the discovery registry.  This matches the commsys design.
 *   - make_node() registers the node in the discovery registry and starts it.
 *     spin_once() sends a heartbeat, polls for new publishers, and drains all
 *     subscriptions.
 */

#pragma once

#include "shmbridge/registry.hpp"
#include "shmbridge/ring.hpp"
#include "shmbridge/topic.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace shmbridge::ros_compat {

/* ── QoS ──────────────────────────────────────────────────────────────────── */

class QoS {
public:
    explicit QoS(uint32_t depth = 10) : depth_(depth) {}

    /* depth <= 1 → seqlock (keep-latest); depth > 1 → SPSC ring. */
    bool wants_keep_latest() const noexcept { return depth_ <= 1; }
    uint32_t depth() const noexcept { return depth_; }

    QoS& best_effort() noexcept { reliable_ = false; return *this; }
    QoS& reliable()    noexcept { reliable_ = true;  return *this; }
    bool is_reliable() const noexcept { return reliable_; }

private:
    uint32_t depth_    = 10;
    bool     reliable_ = true;
};

inline QoS SensorDataQoS()     { return QoS(1).best_effort(); }
inline QoS SystemDefaultsQoS() { return QoS(10).reliable();   }

/* ── Subscription handle (stateless, user-visible) ───────────────────────── */

/*
 * Unlike rclcpp::Subscription, this is a thin handle.  The Node owns the
 * actual subscriber state; this object just carries the topic name so the
 * user can identify the subscription (e.g. to unsubscribe).
 */
template <typename T>
class Subscription {
public:
    explicit Subscription(std::string topic) : topic_(std::move(topic)) {}
    const std::string& topic() const noexcept { return topic_; }
private:
    std::string topic_;
};

/* ── Publisher handle (stateless, user-visible) ───────────────────────────── */

template <typename T>
class Publisher {
public:
    explicit Publisher(std::string topic, bool keep_latest)
        : topic_(std::move(topic)), keep_latest_(keep_latest) {}
    const std::string& topic()       const noexcept { return topic_; }
    bool               keep_latest() const noexcept { return keep_latest_; }
private:
    std::string topic_;
    bool        keep_latest_;
};

/* ── Internal type-erased subscriber slot ─────────────────────────────────── */

namespace detail {

struct SubSlot {
    std::string             topic;
    bool                    keep_latest;  /* true → seqlock, false → ring */
    bool                    attached{false};
    std::function<void()>   poll_fn;      /* drain or read; calls user cb  */
    std::function<bool()>   try_attach;   /* attempt attach if not yet done */
};

struct PubSlot {
    std::string             topic;
    bool                    keep_latest;
    std::function<void()>   noop;         /* placeholder, pub needs no poll */
};

} /* namespace detail */

/* ── Node ─────────────────────────────────────────────────────────────────── */

class Node {
public:
    explicit Node(std::string name) : name_(std::move(name)) {}
    ~Node() { stop(); }

    Node(const Node&)            = delete;
    Node& operator=(const Node&) = delete;

    const std::string& name() const noexcept { return name_; }

    /* Called by make_node(); do not call directly. */
    bool start() noexcept {
        return registry_.open();
    }

    /* ── create_publisher ─────────────────────────────────────────────────── */

    template <typename T>
    std::shared_ptr<Publisher<T>>
    create_publisher(const std::string& topic, uint32_t depth = 10) {
        return create_publisher<T>(topic, QoS(depth));
    }

    template <typename T>
    std::shared_ptr<Publisher<T>>
    create_publisher(const std::string& topic, const QoS& qos) {
        bool kl = qos.wants_keep_latest();
        if (kl) {
            auto pub = std::make_shared<shmbridge::Publisher<T>>();
            if (pub->open(topic.c_str()) != TopicError::Ok)
                throw std::runtime_error("shmbridge: failed to open publisher for " + topic);
            seqlock_pubs_[topic] = pub;
        } else {
            auto pub = std::make_shared<shmbridge::RingPublisher<T>>();
            if (!pub->open(topic.c_str()))
                throw std::runtime_error("shmbridge: failed to open ring publisher for " + topic);
            ring_pubs_[topic] = pub;
        }
        pub_topics_.push_back(topic);
        refresh_registry();
        return std::make_shared<Publisher<T>>(topic, kl);
    }

    /* Publish a message.  Looks up the internal publisher by topic name. */
    template <typename T>
    void publish(const std::string& topic, const T& msg) {
        auto it = seqlock_pubs_.find(topic);
        if (it != seqlock_pubs_.end()) {
            auto pub = std::static_pointer_cast<shmbridge::Publisher<T>>(it->second);
            pub->write_notify(msg);
            return;
        }
        auto it2 = ring_pubs_.find(topic);
        if (it2 != ring_pubs_.end()) {
            auto pub = std::static_pointer_cast<shmbridge::RingPublisher<T>>(it2->second);
            pub->push(msg);
            return;
        }
    }

    /* ── create_subscription ──────────────────────────────────────────────── */

    template <typename T, typename CB>
    std::shared_ptr<Subscription<T>>
    create_subscription(const std::string& topic, uint32_t depth, CB&& cb) {
        return create_subscription<T>(topic, QoS(depth), std::forward<CB>(cb));
    }

    template <typename T, typename CB>
    std::shared_ptr<Subscription<T>>
    create_subscription(const std::string& topic, const QoS& qos, CB&& cb) {
        detail::SubSlot slot;
        slot.topic        = topic;
        slot.keep_latest  = qos.wants_keep_latest();

        if (slot.keep_latest) {
            /* Seqlock subscriber — lazily attach on spin_once(). */
            auto sub = std::make_shared<shmbridge::Subscriber<T>>();
            auto cb_copy = std::function<void(const T&)>(std::forward<CB>(cb));
            slot.try_attach = [sub, topic]() -> bool {
                return sub->attach(topic.c_str(), 10 /* up to 10 ms per try */) == TopicError::Ok;
            };
            slot.poll_fn = [sub, cb_copy]() {
                if (auto r = sub->read_if_new()) cb_copy(r->value);
            };
            seqlock_subs_[topic] = sub;
        } else {
            /* Ring subscriber — lazily attach on spin_once(). */
            auto sub = std::make_shared<shmbridge::RingSubscriber<T>>();
            auto cb_copy = std::function<void(const T&)>(std::forward<CB>(cb));
            slot.try_attach = [sub, topic]() -> bool {
                return sub->attach(topic.c_str(), 10 /* up to 10 ms per try */);
            };
            slot.poll_fn = [sub, cb_copy]() {
                sub->drain([&](const T& msg) { cb_copy(msg); });
            };
            ring_subs_[topic] = sub;
        }

        sub_topics_.push_back(topic);
        refresh_registry();
        sub_slots_.push_back(std::move(slot));
        return std::make_shared<Subscription<T>>(topic);
    }

    /* ── spin_once ────────────────────────────────────────────────────────── */

    /*
     * 1. Send heartbeat.
     * 2. For each un-attached subscriber, try to attach (publisher may have
     *    appeared since last call).
     * 3. Poll every attached subscriber and fire callbacks.
     */
    void spin_once() {
        if (registry_.is_open())
            registry_.heartbeat(registry_.slot_idx());

        for (auto& slot : sub_slots_) {
            if (!slot.attached) {
                slot.attached = slot.try_attach();
            }
            if (slot.attached) {
                slot.poll_fn();
            }
        }
    }

    void stop() noexcept {
        /* Signal all ring publishers closed before destruction. */
        for (auto& [t, p] : ring_pubs_) (void)t;  /* ring cleanup via dtor */
        registry_.close();
    }

private:
    /* Re-register with updated topic lists after each create_* call. */
    void refresh_registry() {
        if (!registry_.is_open()) return;
        int idx = registry_.slot_idx();
        if (idx < 0) {
            /* First registration. */
            std::vector<std::string> subs_kl, subs_ring;
            for (auto& s : sub_slots_)
                (s.keep_latest ? subs_kl : subs_ring).push_back(s.topic);
            registry_.register_node(name_.c_str(), pub_topics_, subs_kl, subs_ring);
        } else {
            std::vector<std::string> subs_kl, subs_ring;
            for (auto& s : sub_slots_)
                (s.keep_latest ? subs_kl : subs_ring).push_back(s.topic);
            registry_.update_topics(idx, pub_topics_, subs_kl, subs_ring);
        }
    }

    std::string name_;
    DiscoveryRegistry registry_;

    /* Internal publisher storage (type-erased via shared_ptr<void>). */
    std::unordered_map<std::string, std::shared_ptr<void>> seqlock_pubs_;
    std::unordered_map<std::string, std::shared_ptr<void>> ring_pubs_;

    /* Internal subscriber storage (type-erased). */
    std::unordered_map<std::string, std::shared_ptr<void>> seqlock_subs_;
    std::unordered_map<std::string, std::shared_ptr<void>> ring_subs_;

    /* Subscriber slots — drive spin_once(). */
    std::vector<detail::SubSlot> sub_slots_;

    /* Topic lists for registry encoding. */
    std::vector<std::string> pub_topics_;
    std::vector<std::string> sub_topics_;
};

/* ── free functions ───────────────────────────────────────────────────────── */

inline void init(int /*argc*/ = 0, char** /*argv*/ = nullptr) noexcept {}
inline void shutdown() noexcept {}

inline std::shared_ptr<Node> make_node(const std::string& name) {
    auto n = std::make_shared<Node>(name);
    if (!n->start())
        throw std::runtime_error("shmbridge: failed to open discovery registry");
    return n;
}

/*
 * spin() — run until the process receives SIGINT / SIGTERM.
 * The caller is responsible for setting up the signal handler; spin() just
 * calls spin_once() + sleep in a loop, checking the flag each iteration.
 */
inline std::atomic<bool> g_ok{true};  /* set false by signal handler */

inline void spin(std::shared_ptr<Node> node,
                 std::chrono::milliseconds period = std::chrono::milliseconds(10)) {
    while (g_ok.load(std::memory_order_relaxed)) {
        node->spin_once();
        std::this_thread::sleep_for(period);
    }
}

inline void spin_some(std::shared_ptr<Node> node) {
    node->spin_once();
}

inline void spin_for(std::shared_ptr<Node> node,
                     std::chrono::milliseconds duration,
                     std::chrono::milliseconds period = std::chrono::milliseconds(10)) {
    auto deadline = std::chrono::steady_clock::now() + duration;
    while (std::chrono::steady_clock::now() < deadline && g_ok.load()) {
        node->spin_once();
        std::this_thread::sleep_for(period);
    }
}

} /* namespace shmbridge::ros_compat */
