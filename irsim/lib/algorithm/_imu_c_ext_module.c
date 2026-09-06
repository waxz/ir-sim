/*
 * _imu_c_ext_module.c  –  Minimal Python extension stub.
 *
 * Both this file and imu_c_ext.c are compiled together by setuptools so the
 * result is a proper Python extension (.so/.pyd) with the correct platform-
 * and OpenMP-specific flags.  imu_c_integrators.py locates the extension via
 * importlib and loads the C functions through ctypes.
 *
 * No public Python API is exposed; the module exists only to give setuptools
 * a valid PyInit_ entry point for each target platform.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Forward-declare the batch integration entry-points from imu_c_ext.c. */
void bench_euler(int n, double dt,
                 const double *omega, const double *ax, const double *ay,
                 double *out_px, double *out_py, double *out_th);
void bench_midpoint(int n, double dt,
                    const double *omega, const double *ax, const double *ay,
                    double *out_px, double *out_py, double *out_th);
void bench_rk4(int n, double dt,
               const double *omega, const double *ax, const double *ay,
               double *out_px, double *out_py, double *out_th);
void bench_strap(int n, double dt,
                 const double *omega, const double *ax, const double *ay,
                 double *out_px, double *out_py, double *out_th);
void bench_mc_midpoint(int n_trials, int n_steps, double dt,
                       const double *omega, const double *ax, const double *ay,
                       double *rmse_out);

static struct PyModuleDef _mod = {
    PyModuleDef_HEAD_INIT,
    "_imu_c_ext",
    "C+OpenMP IMU integration extension (loaded via ctypes by imu_c_integrators.py).",
    -1,
    NULL
};

PyMODINIT_FUNC
PyInit__imu_c_ext(void)
{
    return PyModule_Create(&_mod);
}
