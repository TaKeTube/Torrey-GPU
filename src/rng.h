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

using RNGr = RNG<Real>;
using RNGi = RNG<int>;

__device__ inline Real random_real(RNGr &rng) {
    return rng();
}

__device__ inline int random_int(int min, int max, RNGr &rng) {
    return static_cast<int>(min + (max - min) * random_real(rng));
}