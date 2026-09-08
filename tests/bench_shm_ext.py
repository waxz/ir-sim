"""
bench_shm_ext.py -- Comprehensive benchmark for ExtShmBridge extended channels.

Measures per-call write/read latency, bandwidth, RPS at target frequencies,
CPU fraction, correctness, and torn-read rate for:
  - 6-axis IMU    (40-byte payload)  @ up to 2 kHz
  - 4-wheel Encoder (40-byte payload) @ up to 2 kHz
  - Point cloud   (16-64 KB … 1 MB)  @ 10 Hz

Usage:
    python tests/bench_shm_ext.py

Results saved to tests/bench_ext_results.json.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root))

from irsim.util.shm_bridge_ext import (  # noqa: E402
    EXT_SHM_SIZE,
    N_MAX_POINTS,
    POINT_STRIDE,
    ExtShmBridge,
)

# ── benchmark constants ────────────────────────────────────────────────────────

N_BENCH = 100_000  # iterations per throughput measurement
N_WARMUP = 2_000  # warm-up iterations (not timed)

PC_SIZES = {
    "1K": 1_024,
    "8K": 8_192,
    "32K": 32_768,
    "64K": 65_536,
}

# ── synthetic data ─────────────────────────────────────────────────────────────

_IMU_ARGS = (0.0, 0.12, -9.81, 0.03, 0.001, -0.002, 0.000, 24.5)
_ENC_POS = (0.0, 0.1, 0.2, 0.3)
_ENC_VEL = (1.0, 1.0, 1.0, 1.0)


def _make_cloud(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((n, 4)).astype(np.float32)
    pts[:, 3] = np.abs(pts[:, 3])  # intensity ≥ 0
    return np.ascontiguousarray(pts)


# ── timing helpers ─────────────────────────────────────────────────────────────


def _measure(fn, n: int = N_BENCH, warmup: int = N_WARMUP) -> dict:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter_ns()
    for _ in range(n):
        fn()
    t1 = time.perf_counter_ns()
    elapsed_ns = t1 - t0
    return {
        "n": n,
        "total_ms": elapsed_ns / 1e6,
        "per_call_ns": elapsed_ns / n,
        "ops_per_s": n / (elapsed_ns * 1e-9),
    }


def _timed_run(fn, target_hz: float, duration_s: float) -> dict:
    """Busy-wait rate-limited run; returns actual Hz and CPU fraction."""
    interval_ns = int(1e9 / target_hz)
    count = 0
    cpu_t0 = time.process_time()
    wall_t0 = time.monotonic_ns()
    deadline_ns = wall_t0 + int(duration_s * 1e9)
    next_ns = wall_t0
    while True:
        now = time.monotonic_ns()
        if now >= deadline_ns:
            break
        if now >= next_ns:
            fn()
            count += 1
            next_ns += interval_ns
    wall_elapsed = (time.monotonic_ns() - wall_t0) * 1e-9
    cpu_elapsed = time.process_time() - cpu_t0
    return {
        "target_hz": target_hz,
        "actual_hz": count / wall_elapsed,
        "count": count,
        "wall_s": wall_elapsed,
        "cpu_s": cpu_elapsed,
        "cpu_pct": cpu_elapsed / wall_elapsed * 100.0,
    }


# ── correctness ────────────────────────────────────────────────────────────────


def check_imu(bridge: ExtShmBridge) -> dict:
    bridge.write_imu(1.23, 0.1, 0.2, 9.8, 0.01, -0.02, 0.03, 36.6)
    r = bridge.read_imu()
    ok = (
        r is not None
        and abs(r["timestamp"] - 1.23) < 1e-9
        and abs(r["accel"][2] - 9.8) < 1e-5
        and abs(r["gyro"][1] - (-0.02)) < 1e-6
        and abs(r["temp"] - 36.6) < 0.001
    )
    return {"passed": ok, "result": r}


def check_encoder(bridge: ExtShmBridge) -> dict:
    pos = (1.1, 2.2, 3.3, 4.4)
    vel = (0.5, 0.6, 0.7, 0.8)
    bridge.write_encoder(9.99, pos, vel)
    r = bridge.read_encoder()
    ok = (
        r is not None
        and abs(r["timestamp"] - 9.99) < 1e-9
        and all(abs(r["position"][i] - pos[i]) < 1e-5 for i in range(4))
        and all(abs(r["velocity"][i] - vel[i]) < 1e-6 for i in range(4))
    )
    return {"passed": ok, "result_ts": r["timestamp"] if r else None}


def check_pointcloud(bridge: ExtShmBridge, n: int) -> dict:
    pts = _make_cloud(n)
    bridge.write_pointcloud(pts, 42.0)
    result = bridge.read_pointcloud()
    if result is None:
        return {"passed": False, "reason": "read returned None"}
    meta, data = result
    ok = (
        meta["num_points"] == n
        and data.shape == (n, 4)
        and np.allclose(data, pts, atol=1e-6)
    )
    return {"passed": ok, "n": n, "max_err": float(np.max(np.abs(data - pts)))}


# ── drop-rate test (multiprocessing) ─────────────────────────────────────────


def _writer_proc(
    shm_name: str,
    shm_size: int,
    target_hz: float,
    duration_s: float,
    result_q: mp.Queue[dict],
) -> None:
    """Writer process: writes IMU at target_hz for duration_s."""
    bridge = ExtShmBridge(shm_name=shm_name, shm_size=shm_size)
    bridge.attach()
    interval_ns = int(1e9 / target_hz)
    count = 0
    t0 = time.monotonic_ns()
    deadline = t0 + int(duration_s * 1e9)
    next_ns = t0
    ts = 0.0
    while time.monotonic_ns() < deadline:
        now = time.monotonic_ns()
        if now >= next_ns:
            bridge.write_imu(ts, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)
            count += 1
            ts += 1.0 / target_hz
            next_ns += interval_ns
    elapsed = (time.monotonic_ns() - t0) * 1e-9
    bridge.close()
    result_q.put({"writes": count, "actual_hz": count / elapsed})


def _reader_proc(
    shm_name: str,
    shm_size: int,
    target_hz: float,
    duration_s: float,
    result_q: mp.Queue[dict],
) -> None:
    """Reader process: reads IMU at target_hz, tracks drops via seq2."""
    bridge = ExtShmBridge(shm_name=shm_name, shm_size=shm_size)
    bridge.attach()
    interval_ns = int(1e9 / target_hz)
    reads_ok = 0
    drops = 0
    last_seq2 = 0
    t0 = time.monotonic_ns()
    deadline = t0 + int(duration_s * 1e9)
    next_ns = t0
    while time.monotonic_ns() < deadline:
        now = time.monotonic_ns()
        if now >= next_ns:
            r = bridge.read_imu()
            if r is not None:
                slot = bridge._imu_slot
                cur_seq2 = slot.seq2
                if last_seq2 > 0:
                    frames_since = (cur_seq2 - last_seq2) // 2
                    if frames_since > 1:
                        drops += frames_since - 1
                last_seq2 = cur_seq2
                reads_ok += 1
            next_ns += interval_ns
    elapsed = (time.monotonic_ns() - t0) * 1e-9
    slot = None  # release ctypes ref before closing mmap
    bridge.close()
    result_q.put(
        {"reads_ok": reads_ok, "drops": drops, "actual_hz": reads_ok / elapsed}
    )


def run_drop_rate_test(
    shm_name: str, writer_hz: float, reader_hz: float, duration_s: float = 3.0
) -> dict:
    """
    Spawn writer and reader processes, measure drop rate.
    The owner bridge must already be open (segment exists).
    """
    q: mp.Queue[dict] = mp.Queue()
    wp = mp.Process(
        target=_writer_proc, args=(shm_name, EXT_SHM_SIZE, writer_hz, duration_s, q)
    )
    rp = mp.Process(
        target=_reader_proc, args=(shm_name, EXT_SHM_SIZE, reader_hz, duration_s, q)
    )
    wp.start()
    rp.start()
    wp.join(timeout=duration_s + 5)
    rp.join(timeout=duration_s + 5)
    results = [q.get() for _ in range(2)]
    # sort by which has "writes" key
    writer_r = next(r for r in results if "writes" in r)
    reader_r = next(r for r in results if "reads_ok" in r)
    total_written = writer_r["writes"]
    total_dropped = reader_r["drops"]
    total_seen = reader_r["reads_ok"]
    drop_rate = total_dropped / max(total_written, 1) * 100.0
    return {
        "writer_hz": writer_hz,
        "reader_hz": reader_hz,
        "total_written": total_written,
        "total_seen": total_seen,
        "total_dropped": total_dropped,
        "drop_rate_pct": round(drop_rate, 2),
        "actual_writer_hz": round(writer_r["actual_hz"], 1),
        "actual_reader_hz": round(reader_r["actual_hz"], 1),
    }


# ── torn-read rate ─────────────────────────────────────────────────────────────


def measure_torn_read_rate(bridge: ExtShmBridge, n: int = 200_000) -> dict:
    """
    Write + read in tight alternation; count how often read_imu() returns None
    (seq is odd = writer in progress).  With a single thread this should be 0,
    but measures the seqlock overhead.
    """
    bridge.write_imu(0.0, 0.0, 0.0, 9.81, 0.0, 0.0, 0.0)  # prime the slot
    torn = 0
    ok = 0
    for i in range(n):
        bridge.write_imu(i * 0.001, 0.1, 0.2, 9.8, 0.0, 0.0, 0.0)
        r = bridge.read_imu()
        if r is None:
            torn += 1
        else:
            ok += 1
    return {"n": n, "torn": torn, "ok": ok, "torn_rate_pct": torn / n * 100.0}


# ── CPU combined load ──────────────────────────────────────────────────────────


def combined_load_test(
    bridge: ExtShmBridge,
    imu_hz: float = 1000.0,
    enc_hz: float = 1000.0,
    pc_hz: float = 10.0,
    duration_s: float = 3.0,
) -> dict:
    """
    Write all channels simultaneously at target rates, measure CPU fraction.
    Uses independent rate-limiting loops interleaved in a single thread.
    """
    cloud = _make_cloud(65_536)  # pre-allocate to avoid alloc noise
    imu_interval = int(1e9 / imu_hz)
    enc_interval = int(1e9 / enc_hz)
    pc_interval = int(1e9 / pc_hz)

    n_imu = n_enc = n_pc = 0
    next_imu = next_enc = next_pc = time.monotonic_ns()

    cpu_t0 = time.process_time()
    wall_t0 = time.monotonic_ns()
    deadline = wall_t0 + int(duration_s * 1e9)

    t = 0.0
    while True:
        now = time.monotonic_ns()
        if now >= deadline:
            break
        if now >= next_imu:
            bridge.write_imu(t, 0.12, -9.81, 0.03, 0.001, -0.002, 0.0)
            n_imu += 1
            next_imu += imu_interval
        if now >= next_enc:
            bridge.write_encoder(t, (0.0, 0.1, 0.2, 0.3), (1.0, 1.0, 1.0, 1.0))
            n_enc += 1
            next_enc += enc_interval
        if now >= next_pc:
            bridge.write_pointcloud(cloud, t)
            n_pc += 1
            next_pc += pc_interval
        t += 1e-4  # simulate time advance

    wall_elapsed = (time.monotonic_ns() - wall_t0) * 1e-9
    cpu_elapsed = time.process_time() - cpu_t0
    pc_bandwidth = n_pc * 65_536 * POINT_STRIDE / wall_elapsed / 1e6

    return {
        "imu_actual_hz": round(n_imu / wall_elapsed, 1),
        "enc_actual_hz": round(n_enc / wall_elapsed, 1),
        "pc_actual_hz": round(n_pc / wall_elapsed, 2),
        "pc_bandwidth_mbs": round(pc_bandwidth, 2),
        "cpu_pct": round(cpu_elapsed / wall_elapsed * 100.0, 2),
        "wall_s": round(wall_elapsed, 2),
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Python {sys.version}")
    print(
        f"N_MAX_POINTS={N_MAX_POINTS:,}  PC_DATA_BYTES={N_MAX_POINTS * POINT_STRIDE / 1024:.0f} KiB"
    )
    print(f"EXT_SHM_SIZE={EXT_SHM_SIZE / 1024:.1f} KiB\n")

    pid = os.getpid()
    shm_name = f"/irsim_bench_ext_{pid}"

    with ExtShmBridge(shm_name=shm_name) as bridge:
        # ── 1. Write latency benchmarks ─────────────────────────────────────
        print("=== Write latency (max throughput) ===")
        cloud_1k = _make_cloud(PC_SIZES["1K"])
        cloud_8k = _make_cloud(PC_SIZES["8K"])
        cloud_32k = _make_cloud(PC_SIZES["32K"])
        cloud_64k = _make_cloud(PC_SIZES["64K"])
        t_s = 0.0

        r_imu = _measure(lambda: bridge.write_imu(*_IMU_ARGS))
        r_enc = _measure(lambda: bridge.write_encoder(t_s, _ENC_POS, _ENC_VEL))
        r_pc1k = _measure(
            lambda: bridge.write_pointcloud(cloud_1k, t_s), n=20_000, warmup=500
        )
        r_pc8k = _measure(
            lambda: bridge.write_pointcloud(cloud_8k, t_s), n=10_000, warmup=200
        )
        r_pc32k = _measure(
            lambda: bridge.write_pointcloud(cloud_32k, t_s), n=5_000, warmup=100
        )
        r_pc64k = _measure(
            lambda: bridge.write_pointcloud(cloud_64k, t_s), n=2_000, warmup=50
        )

        # bandwidth for point clouds
        def _bw(r_dict, n_pts):
            return n_pts * POINT_STRIDE / (r_dict["per_call_ns"] * 1e-9) / 1e6

        fmt = "  {:<28} {:>10.1f} ns   {:>8.2f} Mops/s   {:>8.2f} MB/s"
        print(
            fmt.format(
                "write_imu()",
                r_imu["per_call_ns"],
                r_imu["ops_per_s"] / 1e6,
                _bw(r_imu, 5),
            )
        )
        print(
            fmt.format(
                "write_encoder()",
                r_enc["per_call_ns"],
                r_enc["ops_per_s"] / 1e6,
                _bw(r_enc, 5),
            )
        )
        print(
            fmt.format(
                "write_pointcloud(1K pts)",
                r_pc1k["per_call_ns"],
                r_pc1k["ops_per_s"] / 1e6,
                _bw(r_pc1k, PC_SIZES["1K"]),
            )
        )
        print(
            fmt.format(
                "write_pointcloud(8K pts)",
                r_pc8k["per_call_ns"],
                r_pc8k["ops_per_s"] / 1e6,
                _bw(r_pc8k, PC_SIZES["8K"]),
            )
        )
        print(
            fmt.format(
                "write_pointcloud(32K pts)",
                r_pc32k["per_call_ns"],
                r_pc32k["ops_per_s"] / 1e6,
                _bw(r_pc32k, PC_SIZES["32K"]),
            )
        )
        print(
            fmt.format(
                "write_pointcloud(64K pts)",
                r_pc64k["per_call_ns"],
                r_pc64k["ops_per_s"] / 1e6,
                _bw(r_pc64k, PC_SIZES["64K"]),
            )
        )

        # ── 2. Read latency benchmarks ──────────────────────────────────────
        print("\n=== Read latency (max throughput) ===")
        # prime the slots with valid data before reading
        bridge.write_imu(*_IMU_ARGS)
        bridge.write_encoder(0.0, _ENC_POS, _ENC_VEL)
        bridge.write_pointcloud(cloud_1k, 0.0)

        r_rimu = _measure(bridge.read_imu)
        r_renc = _measure(bridge.read_encoder)
        r_rpc_h = _measure(bridge.read_pointcloud_header)
        r_rpc1k = _measure(lambda: bridge.read_pointcloud(), n=20_000, warmup=500)

        fmt2 = "  {:<28} {:>10.1f} ns   {:>8.2f} Mops/s"
        print(
            fmt2.format("read_imu()", r_rimu["per_call_ns"], r_rimu["ops_per_s"] / 1e6)
        )
        print(
            fmt2.format(
                "read_encoder()", r_renc["per_call_ns"], r_renc["ops_per_s"] / 1e6
            )
        )
        print(
            fmt2.format(
                "read_pointcloud_header()",
                r_rpc_h["per_call_ns"],
                r_rpc_h["ops_per_s"] / 1e6,
            )
        )
        print(
            fmt2.format(
                "read_pointcloud(1K pts)",
                r_rpc1k["per_call_ns"],
                r_rpc1k["ops_per_s"] / 1e6,
            )
        )

        # ── 3. Correctness check ─────────────────────────────────────────────
        print("\n=== Correctness ===")
        c_imu = check_imu(bridge)
        c_enc = check_encoder(bridge)
        c_pc1 = check_pointcloud(bridge, 1_024)
        c_pc8 = check_pointcloud(bridge, 8_192)
        c_pc64 = check_pointcloud(bridge, 65_536)
        for name, r in [
            ("IMU roundtrip", c_imu),
            ("Encoder roundtrip", c_enc),
            ("PC 1K roundtrip", c_pc1),
            ("PC 8K roundtrip", c_pc8),
            ("PC 64K roundtrip", c_pc64),
        ]:
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  {name:<28} {status}")

        # ── 4. Torn-read rate (single thread) ───────────────────────────────
        print("\n=== Torn-read rate (single thread, 200K write+read) ===")
        torn = measure_torn_read_rate(bridge)
        print(
            f"  Torn reads: {torn['torn']} / {torn['n']}  ({torn['torn_rate_pct']:.4f}%)"
        )

        # ── 5. Timed rate tests ──────────────────────────────────────────────
        print("\n=== Timed rate (busy-wait, 2s per channel) ===")
        imu_1k = _timed_run(
            lambda: bridge.write_imu(*_IMU_ARGS), target_hz=1000.0, duration_s=2.0
        )
        enc_1k = _timed_run(
            lambda: bridge.write_encoder(0.0, _ENC_POS, _ENC_VEL),
            target_hz=1000.0,
            duration_s=2.0,
        )
        pc_10 = _timed_run(
            lambda: bridge.write_pointcloud(cloud_64k, 0.0),
            target_hz=10.0,
            duration_s=2.0,
        )
        imu_2k = _timed_run(
            lambda: bridge.write_imu(*_IMU_ARGS), target_hz=2000.0, duration_s=2.0
        )

        fmt3 = "  {:<34} actual={:>7.1f} Hz  cpu={:>5.2f}%"
        print(fmt3.format("IMU  1 kHz", imu_1k["actual_hz"], imu_1k["cpu_pct"]))
        print(fmt3.format("IMU  2 kHz", imu_2k["actual_hz"], imu_2k["cpu_pct"]))
        print(fmt3.format("Encoder 1 kHz", enc_1k["actual_hz"], enc_1k["cpu_pct"]))
        print(
            fmt3.format(
                "PointCloud (64K pts) 10 Hz", pc_10["actual_hz"], pc_10["cpu_pct"]
            )
        )

        # ── 6. Drop-rate test (multiprocessing) ─────────────────────────────
        print("\n=== Drop-rate (multiprocessing, 3s each) ===")
        dr1 = run_drop_rate_test(
            shm_name, writer_hz=1000.0, reader_hz=2000.0, duration_s=3.0
        )
        dr2 = run_drop_rate_test(
            shm_name, writer_hz=1000.0, reader_hz=1000.0, duration_s=3.0
        )
        dr3 = run_drop_rate_test(
            shm_name, writer_hz=1000.0, reader_hz=500.0, duration_s=3.0
        )
        fmt4 = "  writer={:>5.0f} Hz  reader={:>5.0f} Hz  written={:>6}  drops={:>5}  drop%={:>5.1f}"
        for dr in [dr1, dr2, dr3]:
            print(
                fmt4.format(
                    dr["writer_hz"],
                    dr["reader_hz"],
                    dr["total_written"],
                    dr["total_dropped"],
                    dr["drop_rate_pct"],
                )
            )

        # ── 7. Combined load ─────────────────────────────────────────────────
        print("\n=== Combined load (IMU 1kHz + Enc 1kHz + PC 10Hz, 3s) ===")
        combined = combined_load_test(
            bridge, imu_hz=1000, enc_hz=1000, pc_hz=10, duration_s=3.0
        )
        print(f"  IMU:     {combined['imu_actual_hz']:.1f} Hz")
        print(f"  Encoder: {combined['enc_actual_hz']:.1f} Hz")
        print(
            f"  PC:      {combined['pc_actual_hz']:.2f} Hz  "
            f"({combined['pc_bandwidth_mbs']:.1f} MB/s)"
        )
        print(
            f"  CPU:     {combined['cpu_pct']:.2f}%  (wall {combined['wall_s']:.2f}s)"
        )

    # ── Save results ─────────────────────────────────────────────────────────
    out = Path(__file__).parent / "bench_ext_results.json"
    results = {
        "write_latency": {
            "imu": r_imu,
            "encoder": r_enc,
            "pc_1k": r_pc1k,
            "pc_8k": r_pc8k,
            "pc_32k": r_pc32k,
            "pc_64k": r_pc64k,
        },
        "write_bandwidth_mbs": {
            "imu": _bw(r_imu, 5),
            "encoder": _bw(r_enc, 5),
            "pc_1k": _bw(r_pc1k, PC_SIZES["1K"]),
            "pc_8k": _bw(r_pc8k, PC_SIZES["8K"]),
            "pc_32k": _bw(r_pc32k, PC_SIZES["32K"]),
            "pc_64k": _bw(r_pc64k, PC_SIZES["64K"]),
        },
        "read_latency": {
            "imu": r_rimu,
            "encoder": r_renc,
            "pc_header": r_rpc_h,
            "pc_1k_copy": r_rpc1k,
        },
        "correctness": {
            "imu": c_imu["passed"],
            "encoder": c_enc["passed"],
            "pc_1k": c_pc1["passed"],
            "pc_8k": c_pc8["passed"],
            "pc_64k": c_pc64["passed"],
            "pc_64k_max_err": c_pc64.get("max_err"),
        },
        "torn_read": torn,
        "timed_rate": {
            "imu_1khz": imu_1k,
            "imu_2khz": imu_2k,
            "encoder_1khz": enc_1k,
            "pc_10hz_64k": pc_10,
        },
        "drop_rate": {
            "writer1k_reader2k": dr1,
            "writer1k_reader1k": dr2,
            "writer1k_reader500": dr3,
        },
        "combined_load": combined,
    }
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
