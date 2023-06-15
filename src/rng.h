#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

template<typename T>
struct RNG
{
    __device__
    RNG(unsigned int tid) {
        curand_init(tid, tid, 0, &state);
    }

    __device__ T
    operator ()(void) {
        return curand_uniform(&state);
    }
    curandState state;
};

using RNGf = RNG<float>;
using RNGi = RNG<int>;

__host__ __device__ inline double random_double(RNGf &rng) {
    return rng();
}

__host__ __device__ inline int random_int(int min, int max, RNGf &rng) {
    return static_cast<int>(min + (max - min) * random_double(rng));
}