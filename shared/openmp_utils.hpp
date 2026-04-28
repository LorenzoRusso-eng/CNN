#pragma once

// OpenMP pragmas are enabled only when the project is compiled with OpenMP support.
// Without the compiler flag these macros become no-ops, so the code stays buildable.

#if defined(_OPENMP)
#include <omp.h>
#endif

#define NN_OMP_STRINGIFY_IMPL(value) #value
#define NN_OMP_STRINGIFY(value) NN_OMP_STRINGIFY_IMPL(value)

#if defined(_OPENMP)
#define NN_OMP_PRAGMA(value) _Pragma(NN_OMP_STRINGIFY(value))
#define NN_OMP_PARALLEL NN_OMP_PRAGMA(omp parallel)
#define NN_OMP_PARALLEL_IF(condition) NN_OMP_PRAGMA(omp parallel if(condition))
#define NN_OMP_FOR NN_OMP_PRAGMA(omp for schedule(static))
#define NN_OMP_FOR_SIMD NN_OMP_PRAGMA(omp for simd schedule(static))
#define NN_OMP_FOR_COLLAPSE(levels) NN_OMP_PRAGMA(omp for collapse(levels) schedule(static))
#define NN_OMP_PARALLEL_FOR NN_OMP_PRAGMA(omp parallel for schedule(static))
#define NN_OMP_PARALLEL_FOR_IF(condition) NN_OMP_PRAGMA(omp parallel for schedule(static) if(condition))
#define NN_OMP_PARALLEL_FOR_COLLAPSE(levels) NN_OMP_PRAGMA(omp parallel for collapse(levels) schedule(static))
#define NN_OMP_PARALLEL_FOR_COLLAPSE_IF(levels, condition) NN_OMP_PRAGMA(omp parallel for collapse(levels) schedule(static) if(condition))
#define NN_OMP_SIMD NN_OMP_PRAGMA(omp simd)
#define NN_OMP_SIMD_REDUCTION_PLUS(var) NN_OMP_PRAGMA(omp simd reduction(+:var))
#define NN_OMP_SIMD_REDUCTION_MAX(var) NN_OMP_PRAGMA(omp simd reduction(max:var))
#else
#define NN_OMP_PARALLEL
#define NN_OMP_PARALLEL_IF(condition)
#define NN_OMP_FOR
#define NN_OMP_FOR_SIMD
#define NN_OMP_FOR_COLLAPSE(levels)
#define NN_OMP_PARALLEL_FOR
#define NN_OMP_PARALLEL_FOR_IF(condition)
#define NN_OMP_PARALLEL_FOR_COLLAPSE(levels)
#define NN_OMP_PARALLEL_FOR_COLLAPSE_IF(levels, condition)
#define NN_OMP_SIMD
#define NN_OMP_SIMD_REDUCTION_PLUS(var)
#define NN_OMP_SIMD_REDUCTION_MAX(var)
#endif

#if defined(_OPENMP)
inline int nn_omp_get_max_threads(){
    return omp_get_max_threads();
}

inline int nn_omp_get_thread_num(){
    return omp_get_thread_num();
}

inline int nn_omp_get_num_threads(){
    return omp_get_num_threads();
}
#else
inline int nn_omp_get_max_threads(){
    return 1;
}

inline int nn_omp_get_thread_num(){
    return 0;
}

inline int nn_omp_get_num_threads(){
    return 1;
}
#endif
