/*
 * Minimal C implementation of the four IMU integration algorithms.
 * Compiled as a shared library and called via ctypes for benchmarking.
 *
 * Demonstrates the speedup achievable with compiled code vs Python/NumPy.
 * The core loop has no memory allocation — all state is maintained in the
 * caller-managed structs passed by pointer.
 *
 * Build:
 *   gcc -O3 -march=native -ffast-math -shared -fPIC -o imu_c_ext.so imu_c_ext.c -lm
 *   (OpenMP variant: add -fopenmp)
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── 2D rotation helper ───────────────────────────────────────────────────── */
static inline void rot2d(double th, double ax, double ay, double *wx, double *wy) {
    double c = cos(th), s = sin(th);
    *wx = c * ax - s * ay;
    *wy = s * ax + c * ay;
}

/* ── State struct shared by all integrators ──────────────────────────────── */
typedef struct {
    double px, py;   /* position */
    double vx, vy;   /* velocity */
    double theta;    /* heading  */
} State;

/* ── Euler (1st order) ────────────────────────────────────────────────────── */
void euler_update(State *st, double omega, double ax, double ay, double dt) {
    double awx, awy;
    rot2d(st->theta, ax, ay, &awx, &awy);
    st->px    += st->vx * dt;
    st->py    += st->vy * dt;
    st->vx    += awx * dt;
    st->vy    += awy * dt;
    st->theta += omega * dt;
}

/* ── Midpoint (2nd order) ─────────────────────────────────────────────────── */
void midpoint_update(State *st, double omega, double ax, double ay, double dt) {
    double theta_mid = st->theta + 0.5 * omega * dt;
    st->theta += omega * dt;
    double awx, awy;
    rot2d(theta_mid, ax, ay, &awx, &awy);
    st->vx += awx * dt;
    st->vy += awy * dt;
    st->px += st->vx * dt;
    st->py += st->vy * dt;
}

/* ── RK4 (4th order) ──────────────────────────────────────────────────────── */
void rk4_update(State *st, double omega, double ax, double ay, double dt) {
    /* state vector: [px, py, vx, vy, theta] */
    double x[5] = {st->px, st->py, st->vx, st->vy, st->theta};
    double k[4][5];

    for (int stage = 0; stage < 4; stage++) {
        double *xp = (stage == 0) ? x : (double[5]){0};
        double alpha = (stage == 0 || stage == 3) ? 1.0 : 0.5;
        double xt[5];

        if (stage > 0) {
            for (int i = 0; i < 5; i++)
                xt[i] = x[i] + alpha * dt * k[stage - 1][i];
            xp = xt;
        }

        double awx, awy;
        rot2d(xp[4], ax, ay, &awx, &awy);
        k[stage][0] = xp[2];
        k[stage][1] = xp[3];
        k[stage][2] = awx;
        k[stage][3] = awy;
        k[stage][4] = omega;
    }

    double coef[4] = {1.0, 2.0, 2.0, 1.0};
    for (int i = 0; i < 5; i++) {
        double sum = 0.0;
        for (int s = 0; s < 4; s++) sum += coef[s] * k[s][i];
        x[i] += (dt / 6.0) * sum;
    }
    st->px = x[0]; st->py = x[1];
    st->vx = x[2]; st->vy = x[3];
    st->theta = x[4];
}

/* ── Strapdown + sculling (2nd order + correction) ───────────────────────── */
typedef struct {
    State base;
    double prev_alpha_x, prev_alpha_y;
    double prev_phi;
} StrapState;

void strap_update(StrapState *st, double omega, double ax, double ay, double dt) {
    double alpha_x = ax * dt, alpha_y = ay * dt;
    double phi = omega * dt;

    /* 2D cross product: phi_scalar x alpha_vec = phi * [-ay, ax] */
    double scull_x = 0.5 * (st->prev_phi * (-alpha_y)   + phi * (-st->prev_alpha_y));
    double scull_y = 0.5 * (st->prev_phi *   alpha_x    + phi *   st->prev_alpha_x);

    double dv_x = alpha_x + scull_x;
    double dv_y = alpha_y + scull_y;

    double theta_mid = st->base.theta + 0.5 * phi;
    st->base.theta += phi;

    double awx, awy;
    rot2d(theta_mid, dv_x / dt, dv_y / dt, &awx, &awy);

    st->base.vx += awx * dt;
    st->base.vy += awy * dt;
    st->base.px += st->base.vx * dt;
    st->base.py += st->base.vy * dt;

    st->prev_alpha_x = alpha_x;
    st->prev_alpha_y = alpha_y;
    st->prev_phi = phi;
}

/*
 * ── Batch benchmark entry-points ──────────────────────────────────────────
 *
 * Each function runs N steps of the given algorithm on pre-computed IMU data
 * arrays (omega[N], ax[N], ay[N]) and writes pose output to out[N*3].
 * These are called from Python via ctypes for timing comparison.
 */

void bench_euler(int n, double dt,
                 const double *omega, const double *ax, const double *ay,
                 double *out_px, double *out_py, double *out_th) {
    State st = {0};
    for (int i = 0; i < n; i++) {
        euler_update(&st, omega[i], ax[i], ay[i], dt);
        out_px[i] = st.px;
        out_py[i] = st.py;
        out_th[i] = st.theta;
    }
}

void bench_midpoint(int n, double dt,
                    const double *omega, const double *ax, const double *ay,
                    double *out_px, double *out_py, double *out_th) {
    State st = {0};
    for (int i = 0; i < n; i++) {
        midpoint_update(&st, omega[i], ax[i], ay[i], dt);
        out_px[i] = st.px;
        out_py[i] = st.py;
        out_th[i] = st.theta;
    }
}

void bench_rk4(int n, double dt,
               const double *omega, const double *ax, const double *ay,
               double *out_px, double *out_py, double *out_th) {
    State st = {0};
    for (int i = 0; i < n; i++) {
        rk4_update(&st, omega[i], ax[i], ay[i], dt);
        out_px[i] = st.px;
        out_py[i] = st.py;
        out_th[i] = st.theta;
    }
}

void bench_strap(int n, double dt,
                 const double *omega, const double *ax, const double *ay,
                 double *out_px, double *out_py, double *out_th) {
    StrapState st;
    memset(&st, 0, sizeof(st));
    for (int i = 0; i < n; i++) {
        strap_update(&st, omega[i], ax[i], ay[i], dt);
        out_px[i] = st.base.px;
        out_py[i] = st.base.py;
        out_th[i] = st.base.theta;
    }
}

/*
 * ── OpenMP Monte-Carlo benchmark ──────────────────────────────────────────
 *
 * Runs n_trials independent midpoint integrations in parallel using OpenMP.
 * Each trial uses the same IMU input; in a real implementation each trial
 * would use independent noise samples.
 *
 * Compile with -fopenmp to enable; without it the loop runs serially.
 */
#ifdef _OPENMP
#include <omp.h>
#endif

void bench_mc_midpoint(int n_trials, int n_steps, double dt,
                       const double *omega, const double *ax, const double *ay,
                       double *rmse_out) {
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int trial = 0; trial < n_trials; trial++) {
        State st = {0};
        double sq_sum = 0.0;
        for (int i = 0; i < n_steps; i++) {
            midpoint_update(&st, omega[i], ax[i], ay[i], dt);
            sq_sum += st.px * st.px + st.py * st.py;
        }
        rmse_out[trial] = sqrt(sq_sum / n_steps);
    }
}
