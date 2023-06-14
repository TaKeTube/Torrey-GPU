#pragma once
// #ifndef TORREY_H
// #define TORREY_H

// CMake insert NDEBUG when building with RelWithDebInfo
// This is an ugly hack to undo that...
#undef NDEBUG

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <algorithm>
#include <random>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<cuda.h>

// for suppressing unused warnings
#define UNUSED(x) (void)(x)

// We use double for most of our computation.
// Rendering is usually done in single precision Reals.
// However, torrey is an educational renderer with does not
// put emphasis on the absolute performance. 
// We choose double so that we do not need to worry about
// numerical accuracy as much when we render.
// Switching to floating point computation is easy --
// just set Real = float.
using Real = double;

#define c_EPSILON 1e-7

// Lots of PIs!
#define c_PI 3.14159265358979323846
#define c_INVPI = (1.0 / c_PI)
#define c_TWOPI = (2.0 * c_PI)
#define c_INVTWOPI = (1.0 / c_TWOPI)
#define c_FOURPI = (4.0 * c_PI)
#define c_INVFOURPI = (1.0 / c_FOURPI)
#define c_PIOVERTWO = (0.5 * c_PI)
#define c_PIOVERFOUR = (0.25 * c_PI)

template <typename T>
inline T infinity() {
    return std::numeric_limits<T>::infinity();
}

namespace fs = std::filesystem;

inline std::string to_lowercase(const std::string &s) {
    std::string out = s;
    std::transform(s.begin(), s.end(), out.begin(), ::tolower);
    return out;
}

__host__ __device__ inline int modulo(int a, int b) {
    auto r = a % b;
    return (r < 0) ? r+b : r;
}

__host__ __device__ inline float modulo(float a, float b) {
    float r = ::fmodf(a, b);
    return (r < 0.0f) ? r+b : r;
}

__host__ __device__ inline double modulo(double a, double b) {
    double r = ::fmod(a, b);
    return (r < 0.0) ? r+b : r;
}

template <typename T>
__host__ __device__ inline T max(const T &a, const T &b) {
    return a > b ? a : b;
}

template <typename T>
__host__ __device__ inline T min(const T &a, const T &b) {
    return a < b ? a : b;
}

__host__ __device__ inline Real radians(const Real deg) {
    return (c_PI / Real(180)) * deg;
}

__host__ __device__ inline Real degrees(const Real rad) {
    return (Real(180) / c_PI) * rad;
}

inline double random_double(std::mt19937 &rng) {
    return std::uniform_real_distribution<double>{0.0, 1.0}(rng);
}

inline int random_int(int min, int max, std::mt19937 &rng) {
    return static_cast<int>(min + (max - min) * random_double(rng));
}

// #endif