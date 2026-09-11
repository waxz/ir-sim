/*
 * ray_casting_omp.c - OpenMP-parallel 2D ray-segment intersection kernels.
 *
 * Compile (Linux/macOS — uses native CPU features):
 *   gcc -O3 -march=native -fopenmp -shared -fPIC -o ray_casting_omp.so \
 *       ray_casting_omp.c -lm
 *
 * Compile (explicit AVX2, portable to any AVX2 x86-64 host):
 *   gcc -O3 -mavx2 -fopenmp -shared -fPIC -o ray_casting_omp.so \
 *       ray_casting_omp.c -lm
 *
 * Called from ray_casting_2d_omp.py via ctypes.  The function signature is
 * a plain-C ABI so no Python headers are required.
 *
 * Kernel selection and platform fallback:
 *
 *   Platform           Kernel compiled       Python fallback chain
 *   ─────────────────  ────────────────────  ─────────────────────────────────
 *   x86-64 with AVX2   OMP + AVX2 SoA        AVX2 > OMP > NumPy
 *   x86-64 no AVX2     OMP only              OMP > NumPy
 *   x86-64 Windows     OMP + AVX2 (MSVC      AVX2 > OMP > NumPy
 *                       /arch:AVX2 required)
 *   AArch64 / Apple M  OMP only              OMP > NumPy
 *   Any (no compiler)  (not compiled)        NumPy
 *
 * cast_ray_segments_avx2_soa:
 *   AVX2 SIMD 4-wide, OpenMP group-parallel (SoA layout).
 *   Compiled only on x86/x86-64 with __AVX2__ defined.
 *   Collinear ray-segment overlap (denom==0 && cross≈0) is not handled
 *   in this path — the scalar OMP kernel covers that rare case.
 */

#include <math.h>
#include <stdint.h>
#include <float.h>

/*
 * AVX2 SIMD path — x86/x86-64 only.
 *
 * The compound guard checks both the ISA extension (__AVX2__) *and* the CPU
 * architecture family so that <immintrin.h> is never included on non-x86
 * targets (AArch64, RISC-V, PowerPC, WASM, …).  When the guard is false,
 * cast_ray_segments_avx2_soa is absent from the compiled binary and
 * ray_casting_2d_omp.py falls back to cast_ray_segments_omp automatically.
 *
 * To add NEON support for AArch64 in the future:
 *   #elif defined(__ARM_NEON__) && defined(__aarch64__)
 *   #include <arm_neon.h>
 *   // … 2-wide float64x2_t implementation …
 */
#if defined(__AVX2__) && \
    (defined(__x86_64__) || defined(_M_X64) || \
     defined(__i386__)   || defined(_M_IX86))
#define _IRSIM_AVX2 1
#include <immintrin.h>
#endif

#define ORIGIN_EPS 1e-9

/*
 * cast_ray_segments_omp
 *
 * Parameters (all arrays are row-major / C order):
 *   origin      - double[2]     ray origin
 *   directions  - double[N*2]   unit beam directions (row = [dx, dy])
 *   seg_start   - double[M*2]   segment start points
 *   seg_end     - double[M*2]   segment end points
 *   N           - int           number of beams
 *   M           - int           number of segments
 *   max_range   - double        miss distance
 *   out_ranges  - double[N]     output: hit distances
 *   out_hit     - int64_t[N]    output: hit segment indices (-1 = miss)
 */
void cast_ray_segments_omp(
    const double *origin,
    const double *directions,
    const double *seg_start,
    const double *seg_end,
    int N, int M,
    double max_range,
    double *out_ranges,
    int64_t *out_hit
) {
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 32)
    #endif
    int i;
    for (i = 0; i < N; i++) {
        double dx  = directions[2*i];
        double dy  = directions[2*i + 1];
        double pdx = -dy;   /* perpendicular to beam */
        double pdy =  dx;

        double best_t = max_range + 1.0;  /* sentinel */
        int    best_j = -1;

        for (int j = 0; j < M; j++) {
            double svx = seg_end[2*j]     - seg_start[2*j];
            double svy = seg_end[2*j + 1] - seg_start[2*j + 1];
            double sox = origin[0] - seg_start[2*j];
            double soy = origin[1] - seg_start[2*j + 1];

            /* denom = dot(seg_vec, perp_dir) */
            double denom = svx * pdx + svy * pdy;

            if (denom == 0.0) {
                /* Parallel: check collinear overlap */
                double cross = svx * soy - svy * sox;
                if (fabs(cross) <= ORIGIN_EPS) {
                    double ta = (seg_start[2*j]     - origin[0]) * dx
                              + (seg_start[2*j + 1] - origin[1]) * dy;
                    double tb = (seg_end[2*j]        - origin[0]) * dx
                              + (seg_end[2*j + 1]   - origin[1]) * dy;
                    double ov_start = ta < tb ? ta : tb;
                    double ov_end   = ta < tb ? tb : ta;
                    if (ov_end > ORIGIN_EPS && ov_start <= max_range) {
                        double hit_t = ov_start > ORIGIN_EPS
                            ? ov_start
                            : (ov_end <= max_range ? ov_end : max_range);
                        if (hit_t <= max_range && hit_t < best_t) {
                            best_t = hit_t;
                            best_j = j;
                        }
                    }
                }
                continue;
            }

            double u = (sox * pdx + soy * pdy) / denom;
            if (u < 0.0 || u > 1.0) continue;

            double cross = svx * soy - svy * sox;
            double t = cross / denom;
            if (t > ORIGIN_EPS && t <= max_range && t < best_t) {
                best_t = t;
                best_j = j;
            }
        }

        if (best_j >= 0) {
            out_ranges[i] = best_t;
            out_hit[i]    = (int64_t)best_j;
        } else {
            out_ranges[i] = max_range;
            out_hit[i]    = -1;
        }
    }
}

#ifdef _IRSIM_AVX2
/*
 * cast_ray_segments_avx2_soa
 *
 * AVX2 SIMD variant: processes 4 beams simultaneously using 256-bit
 * double-precision registers.  Uses SoA (Structure-of-Arrays) layout so
 * segment scalars (svx, svy, sox, soy, cross) are broadcast once per
 * segment for all 4 beam lanes.
 *
 * Collinear overlap (denom==0 AND cross≈0) is skipped in this path;
 * use cast_ray_segments_omp for exact handling of that rare case.
 *
 * Parameters:
 *   origin    - double[2]    ray origin (shared by all beams)
 *   dir_dx    - double[N]    beam directions: x component  (SoA)
 *   dir_dy    - double[N]    beam directions: y component  (SoA)
 *   seg_sx    - double[M]    segment start x               (SoA)
 *   seg_sy    - double[M]    segment start y               (SoA)
 *   seg_ex    - double[M]    segment end x                 (SoA)
 *   seg_ey    - double[M]    segment end y                 (SoA)
 *   N         - int          number of beams
 *   M         - int          number of segments
 *   max_range - double       miss distance
 *   out_ranges - double[N]   output: hit distances
 *   out_hit   - int64_t[N]   output: hit segment indices (-1 = miss)
 */
void cast_ray_segments_avx2_soa(
    const double *origin,
    const double *dir_dx,
    const double *dir_dy,
    const double *seg_sx,
    const double *seg_sy,
    const double *seg_ex,
    const double *seg_ey,
    int N, int M,
    double max_range,
    double *out_ranges,
    int64_t *out_hit
) {
    const double ox       = origin[0];
    const double oy       = origin[1];
    const double sentinel = max_range + 1.0;

    /* AVX2 block: process 4 beams per iteration */
    int avx_n = N & ~3;  /* round down to multiple of 4 */

    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 8)
    #endif
    int base;
    for (base = 0; base < avx_n; base += 4) {
        __m256d dx_v  = _mm256_loadu_pd(dir_dx + base);
        __m256d dy_v  = _mm256_loadu_pd(dir_dy + base);
        /* perpendicular to each beam: pdx = -dy, pdy = dx */
        __m256d pdx_v = _mm256_sub_pd(_mm256_setzero_pd(), dy_v);
        __m256d pdy_v = dx_v;

        __m256d best_t_v = _mm256_set1_pd(sentinel);
        __m256i best_j_v = _mm256_set1_epi64x(-1LL);

        const __m256d zero_v = _mm256_setzero_pd();
        const __m256d one_v  = _mm256_set1_pd(1.0);
        const __m256d eps_v  = _mm256_set1_pd(ORIGIN_EPS);
        const __m256d maxr_v = _mm256_set1_pd(max_range);

        for (int j = 0; j < M; j++) {
            /* Segment-derived scalars: identical for all 4 beam lanes */
            double svx   = seg_ex[j] - seg_sx[j];
            double svy   = seg_ey[j] - seg_sy[j];
            double sox   = ox - seg_sx[j];
            double soy   = oy - seg_sy[j];
            double cross = svx * soy - svy * sox;

            __m256d svx_v   = _mm256_set1_pd(svx);
            __m256d svy_v   = _mm256_set1_pd(svy);
            __m256d sox_v   = _mm256_set1_pd(sox);
            __m256d soy_v   = _mm256_set1_pd(soy);
            __m256d cross_v = _mm256_set1_pd(cross);

            /* denom[i] = svx * pdx[i] + svy * pdy[i] */
            __m256d denom_v = _mm256_fmadd_pd(svy_v, pdy_v,
                                  _mm256_mul_pd(svx_v, pdx_v));

            /* u[i] = (sox * pdx[i] + soy * pdy[i]) / denom[i] */
            __m256d u_v = _mm256_div_pd(
                _mm256_fmadd_pd(soy_v, pdy_v, _mm256_mul_pd(sox_v, pdx_v)),
                denom_v);

            /* t[i] = cross / denom[i] */
            __m256d t_v = _mm256_div_pd(cross_v, denom_v);

            /* Acceptance mask: u in [0,1], t in (eps, max_range], t < best_t */
            __m256d mask = _mm256_and_pd(
                _mm256_and_pd(
                    _mm256_and_pd(_mm256_cmp_pd(u_v, zero_v, _CMP_GE_OQ),   /* u >= 0 */
                                  _mm256_cmp_pd(u_v, one_v,  _CMP_LE_OQ)),  /* u <= 1 */
                    _mm256_and_pd(_mm256_cmp_pd(t_v, eps_v,  _CMP_GT_OQ),   /* t > eps */
                                  _mm256_cmp_pd(t_v, maxr_v, _CMP_LE_OQ))), /* t <= max_range */
                _mm256_cmp_pd(t_v, best_t_v, _CMP_LT_OQ));                  /* t < best_t */

            best_t_v = _mm256_blendv_pd(best_t_v, t_v, mask);
            best_j_v = _mm256_blendv_epi8(best_j_v,
                           _mm256_set1_epi64x((long long)j),
                           _mm256_castpd_si256(mask));
        }

        /* Store 4 results */
        double    bt[4];
        long long bj[4];
        _mm256_storeu_pd(bt, best_t_v);
        _mm256_storeu_si256((__m256i *)bj, best_j_v);
        for (int k = 0; k < 4; k++) {
            int idx = base + k;
            if (bj[k] >= 0) {
                out_ranges[idx] = bt[k];
                out_hit[idx]    = (int64_t)bj[k];
            } else {
                out_ranges[idx] = max_range;
                out_hit[idx]    = -1;
            }
        }
    }

    /* Scalar tail: handles beams when N is not a multiple of 4 */
    for (int i = avx_n; i < N; i++) {
        double dx  = dir_dx[i];
        double dy  = dir_dy[i];
        double pdx = -dy;
        double pdy =  dx;
        double best_t = sentinel;
        int    best_j = -1;

        for (int j = 0; j < M; j++) {
            double svx = seg_ex[j] - seg_sx[j];
            double svy = seg_ey[j] - seg_sy[j];
            double sox = ox - seg_sx[j];
            double soy = oy - seg_sy[j];

            double denom = svx * pdx + svy * pdy;
            if (denom == 0.0) continue;

            double u = (sox * pdx + soy * pdy) / denom;
            if (u < 0.0 || u > 1.0) continue;

            double cross = svx * soy - svy * sox;
            double t = cross / denom;
            if (t > ORIGIN_EPS && t <= max_range && t < best_t) {
                best_t = t;
                best_j = j;
            }
        }

        if (best_j >= 0) {
            out_ranges[i] = best_t;
            out_hit[i]    = (int64_t)best_j;
        } else {
            out_ranges[i] = max_range;
            out_hit[i]    = -1;
        }
    }
}
#endif /* _IRSIM_AVX2 */
