/*
 * shmbridge/notify.hpp — pluggable notification strategies for shmbridge v3.
 *
 * INotifier: abstract base (virtual, for runtime polymorphism).
 * SemNotifier: POSIX semaphore — wakes ONE waiter per post (unicast).
 * ShmFutexNotifier: futex on shared memory — one FUTEX_WAKE wakes ALL waiters
 *   simultaneously (broadcast), O(1) publisher cost regardless of subscriber count.
 * NullNotifier: no-op — pure spin / poll, no side channel.
 *
 * Only ShmFutexNotifier is cross-process safe without extra setup.
 * SemNotifier is most portable (macOS included); ShmFutexNotifier is Linux-only.
 */

#pragma once

#include <cstdint>
#include <ctime>

#if defined(__linux__)
#  include <syscall.h>
#  include <linux/futex.h>
#  include <unistd.h>
#  include <climits>
#endif

#if defined(__APPLE__) || defined(__linux__)
#  include <semaphore.h>
#  include <fcntl.h>   /* O_CREAT, O_EXCL */
#  include <cerrno>
#  include <cstring>   /* strerror */
#  include <cstdio>    /* snprintf */
#endif

namespace shmbridge {

/* ── abstract interface ──────────────────────────────────────────────────── */

struct INotifier {
    virtual ~INotifier()              = default;
    virtual void notify()             noexcept = 0;
    virtual void wait(int timeout_ms) noexcept = 0;
};

/* ── no-op notifier (pure-spin subscribers) ──────────────────────────────── */

struct NullNotifier final : INotifier {
    void notify()             noexcept override {}
    void wait(int /*timeout*/) noexcept override {}
};

/* ── POSIX named semaphore (unicast, portable) ───────────────────────────── */

#if defined(__APPLE__) || defined(__linux__)

class SemNotifier final : public INotifier {
public:
    /* Publisher: call open_create; subscriber: call open_attach */
    bool open_create(const char* topic_name) noexcept {
        build_name(topic_name);
        ::sem_unlink(name_);
        sem_ = ::sem_open(name_, O_CREAT | O_EXCL, 0600, 0);
        return sem_ != SEM_FAILED;
    }
    bool open_attach(const char* topic_name, int timeout_ms = 5000) noexcept {
        build_name(topic_name);
        /* Poll until publisher creates the sem */
        const long deadline_ns = static_cast<long>(timeout_ms) * 1'000'000L;
        long waited = 0;
        while (waited < deadline_ns) {
            sem_ = ::sem_open(name_, 0);
            if (sem_ != SEM_FAILED) return true;
            struct timespec sl{0, 5'000'000L};
            ::nanosleep(&sl, nullptr);
            waited += 5'000'000L;
        }
        return false;
    }
    void close() noexcept {
        if (sem_ && sem_ != SEM_FAILED) { ::sem_close(sem_); sem_ = nullptr; }
    }
    void unlink() noexcept { ::sem_unlink(name_); }

    void notify() noexcept override {
        if (sem_ && sem_ != SEM_FAILED) ::sem_post(sem_);
    }
    void wait(int timeout_ms) noexcept override {
        if (!sem_ || sem_ == SEM_FAILED) return;
        if (timeout_ms < 0) {
            ::sem_wait(sem_);
        } else {
#if defined(__linux__)
            struct timespec abs{};
            ::clock_gettime(CLOCK_REALTIME, &abs);
            abs.tv_sec  += timeout_ms / 1000;
            abs.tv_nsec += (long)(timeout_ms % 1000) * 1'000'000L;
            if (abs.tv_nsec >= 1'000'000'000L) { abs.tv_sec++; abs.tv_nsec -= 1'000'000'000L; }
            ::sem_timedwait(sem_, &abs);
#else
            /* macOS lacks sem_timedwait — busy-poll */
            long waited = 0, deadline = (long)timeout_ms * 1'000'000L;
            while (waited < deadline) {
                if (::sem_trywait(sem_) == 0) return;
                struct timespec sl{0, 500'000L};
                ::nanosleep(&sl, nullptr);
                waited += 500'000L;
            }
#endif
        }
    }

private:
    void build_name(const char* t) noexcept {
        ::snprintf(name_, sizeof(name_), "/sbsem_%.56s", t);
    }
    sem_t* sem_  = nullptr;
    char   name_[64]{};
};

#endif /* POSIX */

/* ── futex-based broadcast notifier (Linux, cross-process) ──────────────── */

#if defined(__linux__)

/*
 * Uses a `volatile uint32_t` word in shared memory as the futex.
 * Publisher increments the word and calls FUTEX_WAKE with INT_MAX —
 * one syscall wakes every subscriber regardless of how many there are.
 *
 * Usage:
 *   // in TopicHeader (shared memory, already mmap'd):
 *   volatile uint32_t notify_seq = 0;
 *
 *   ShmFutexNotifier pub_n, sub_n;
 *   pub_n.bind(&header->notify_seq);
 *   sub_n.bind(&header->notify_seq);
 *
 *   pub_n.notify();   // increments + FUTEX_WAKE(INT_MAX) → wakes all
 *   sub_n.wait(100);  // FUTEX_WAIT until changed or timeout
 */
class ShmFutexNotifier final : public INotifier {
public:
    void bind(volatile uint32_t* word) noexcept { word_ = word; }

    void notify() noexcept override {
        if (!word_) return;
        __atomic_fetch_add(const_cast<uint32_t*>(word_), 1u, __ATOMIC_RELEASE);
        ::syscall(SYS_futex, const_cast<uint32_t*>(word_),
                  FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
    }

    void wait(int timeout_ms = -1) noexcept override {
        if (!word_) return;
        uint32_t val = __atomic_load_n(const_cast<const uint32_t*>(word_),
                                       __ATOMIC_ACQUIRE);
        if (timeout_ms < 0) {
            ::syscall(SYS_futex, const_cast<uint32_t*>(word_),
                      FUTEX_WAIT, val, nullptr, nullptr, 0);
        } else {
            struct timespec ts{
                static_cast<time_t>(timeout_ms / 1000),
                static_cast<long>(timeout_ms % 1000) * 1'000'000L
            };
            ::syscall(SYS_futex, const_cast<uint32_t*>(word_),
                      FUTEX_WAIT, val, &ts, nullptr, 0);
        }
    }

private:
    volatile uint32_t* word_ = nullptr;
};

#endif /* __linux__ */

} /* namespace shmbridge */
