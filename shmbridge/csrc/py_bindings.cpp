/*
 * py_bindings.cpp  -  pybind11 Python bindings for shmbridge::ShmPublisher
 *                     and shmbridge::ShmSubscriber.
 *
 * Compiled by scikit-build-core into _core.cpython-*.so and installed
 * alongside the pure-Python shmbridge package.  When present it is loaded
 * by __init__.py and takes precedence over the ctypes fallback.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <shmbridge/core.hpp>

namespace py = pybind11;
using namespace shmbridge;

PYBIND11_MODULE(_core, m) {
    m.doc() = "shmbridge C++ core: ShmPublisher / ShmSubscriber";

    /* ── RobotState ────────────────────────────────────────────────────── */
    py::class_<RobotState>(m, "RobotState")
        .def(py::init<>())
        .def_readwrite("x",         &RobotState::x)
        .def_readwrite("y",         &RobotState::y)
        .def_readwrite("heading",   &RobotState::heading)
        .def_readwrite("vx",        &RobotState::vx)
        .def_readwrite("vy",        &RobotState::vy)
        .def_readwrite("omega",     &RobotState::omega)
        .def_readwrite("goal_x",    &RobotState::goal_x)
        .def_readwrite("goal_y",    &RobotState::goal_y)
        .def_readwrite("goal_dist", &RobotState::goal_dist)
        .def_readwrite("step",      &RobotState::step)
        .def_readwrite("sim_time",  &RobotState::sim_time)
        .def_readwrite("reached",   &RobotState::reached)
        .def_readwrite("collision", &RobotState::collision)
        .def("__repr__", [](const RobotState& s) {
            return "<RobotState x=" + std::to_string(s.x) +
                   " y=" + std::to_string(s.y) +
                   " step=" + std::to_string(s.step) + ">";
        });

    /* ── RobotCmd ──────────────────────────────────────────────────────── */
    py::class_<RobotCmd>(m, "RobotCmd")
        .def(py::init<>())
        .def(py::init([](float lin, float ang, uint32_t seq) {
            return RobotCmd{lin, ang, seq};
        }), py::arg("linear") = 0.f, py::arg("angular") = 0.f,
            py::arg("seq") = 0u)
        .def_readwrite("linear",  &RobotCmd::linear)
        .def_readwrite("angular", &RobotCmd::angular)
        .def_readwrite("seq",     &RobotCmd::seq)
        .def("__repr__", [](const RobotCmd& c) {
            return "<RobotCmd linear=" + std::to_string(c.linear) +
                   " angular=" + std::to_string(c.angular) +
                   " seq=" + std::to_string(c.seq) + ">";
        });

    /* ── ShmPublisher ──────────────────────────────────────────────────── */
    py::class_<ShmPublisher>(m, "ShmPublisher",
        R"doc(
        Creates and owns the shm segment; writes robot state, reads cmds.

        Parameters
        ----------
        name : str
            POSIX shm name (default "/irsim_bridge_v2").
        n_robots : int
            Number of robot slots (default 1).
        n_consumers : int
            Independent cmd writers per robot (default 1).
        heartbeat_every : int
            Update writer_ts_ns every N writes (default 1).
        )doc")
        .def(py::init<std::string, unsigned, unsigned, unsigned>(),
             py::arg("name")            = SHMBRIDGE_SHM_NAME,
             py::arg("n_robots")        = 1u,
             py::arg("n_consumers")     = 1u,
             py::arg("heartbeat_every") = 1u)
        .def("open",    &ShmPublisher::open, py::arg("mlock") = false,
             "Create and zero-init the shm segment.")
        .def("close",   &ShmPublisher::close,
             "Unmap and unlink the segment.")
        .def("is_open", &ShmPublisher::is_open)
        .def("__enter__", [](ShmPublisher& self) -> ShmPublisher& {
            self.open(); return self;
        })
        .def("__exit__", [](ShmPublisher& self, py::object, py::object,
                             py::object) { self.close(); })
        .def("write_state", &ShmPublisher::write_state,
             py::arg("robot_idx"), py::arg("state"),
             "Seqlock-write robot state (fast; GIL released).")
        .def("read_cmd",
             [](const ShmPublisher& self, unsigned r, unsigned c) {
                 return self.read_cmd(r, c);
             },
             py::arg("robot_idx") = 0u, py::arg("consumer_idx") = 0u,
             "Non-blocking seqlock read from one consumer slot. Returns None "
             "if mid-write or no valid cmd.")
        .def("read_best_cmd",
             [](const ShmPublisher& self, unsigned r) {
                 return self.read_best_cmd(r);
             },
             py::arg("robot_idx") = 0u,
             "Return the valid cmd with the highest seq across all consumer "
             "slots, or None.")
        .def("is_controller_alive", &ShmPublisher::is_controller_alive,
             py::arg("max_age_ms")   = 100.0,
             py::arg("robot_idx")    = 0u,
             py::arg("consumer_idx") = 0u)
        .def_property_readonly("n_robots",    &ShmPublisher::n_robots)
        .def_property_readonly("n_consumers", &ShmPublisher::n_consumers);

    /* ── ShmSubscriber ─────────────────────────────────────────────────── */
    py::class_<ShmSubscriber>(m, "ShmSubscriber",
        R"doc(
        Attaches to an existing shm segment; reads state, writes cmds.

        Multiple ShmSubscribers may attach to the same segment simultaneously.
        Each uses a distinct consumer_idx so the publisher can arbitrate via
        read_best_cmd().

        Parameters
        ----------
        name : str
            POSIX shm name (must match the publisher's).
        n_robots : int
            Hint; overridden by the segment header on attach().
        )doc")
        .def(py::init<std::string, unsigned>(),
             py::arg("name")     = SHMBRIDGE_SHM_NAME,
             py::arg("n_robots") = 1u)
        .def("attach", &ShmSubscriber::attach, py::arg("timeout_ms") = 30000u,
             "Attach to an existing segment; blocks until ready.")
        .def("detach",      &ShmSubscriber::detach)
        .def("is_attached", &ShmSubscriber::is_attached)
        .def("__enter__", [](ShmSubscriber& self) -> ShmSubscriber& {
            self.attach(); return self;
        })
        .def("__exit__", [](ShmSubscriber& self, py::object, py::object,
                             py::object) { self.detach(); })
        .def("read_state",
             [](const ShmSubscriber& self, unsigned r) {
                 return self.read_state(r);
             },
             py::arg("robot_idx") = 0u,
             "Seqlock read; returns RobotState or None on torn read.")
        .def("read_state_spin",
             [](const ShmSubscriber& self, unsigned r, unsigned retries) {
                 return self.read_state_spin(r, retries);
             },
             py::arg("robot_idx") = 0u, py::arg("max_retries") = 64u,
             "Spin until a clean read; returns None only on persistent torn reads.")
        .def("write_cmd",
             [](ShmSubscriber& self, unsigned r, unsigned c,
                float lin, float ang) {
                 self.write_cmd(r, c, lin, ang);
             },
             py::arg("robot_idx"), py::arg("consumer_idx"),
             py::arg("linear"), py::arg("angular"),
             "Seqlock-write a velocity command to the specified consumer slot.")
        .def("is_publisher_alive", &ShmSubscriber::is_publisher_alive,
             py::arg("max_age_ms") = 100.0, py::arg("robot_idx") = 0u)
        .def_property_readonly("n_robots",    &ShmSubscriber::n_robots)
        .def_property_readonly("n_consumers", &ShmSubscriber::n_consumers);
}
