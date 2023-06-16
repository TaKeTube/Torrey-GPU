#pragma once

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
#include <cuda.h>
#include <cuda_runtime.h>

// for suppressing unused warnings
#define UNUSED(x) (void)(x)

#define c_EPSILON 1e-7

// Lots of PIs!
#define c_PI            Real(3.14159265358979323846)
#define c_INVPI         Real(1.0 / c_PI)
#define c_TWOPI         Real(2.0 * c_PI)
#define c_INVTWOPI      Real(1.0 / c_TWOPI)
#define c_FOURPI        Real(4.0 * c_PI)
#define c_INVFOURPI     Real(1.0 / c_FOURPI)
#define c_PIOVERTWO     Real(0.5 * c_PI)
#define c_PIOVERFOUR    Real(0.25 * c_PI)

// We use double for most of our computation.
// Rendering is usually done in single precision Reals.
// However, torrey is an educational renderer with does not
// put emphasis on the absolute performance. 
// We choose double so that we do not need to worry about
// numerical accuracy as much when we render.
// Switching to floating point computation is easy --
// just set Real = float.
using Real = double;
// using Real = float;

constexpr int TILE_WIDTH = 16;
// constexpr int GRID_WIDTH = 512;
constexpr int STACK_SIZE = 64;

// constexpr Real c_EPSILON = 1e-7;

// // Lots of PIs!
// constexpr Real c_PI = Real(3.14159265358979323846);
// constexpr Real c_INVPI = Real(1.0) / c_PI;
// constexpr Real c_TWOPI = Real(2.0) * c_PI;
// constexpr Real c_INVTWOPI = Real(1.0) / c_TWOPI;
// constexpr Real c_FOURPI = Real(4.0) * c_PI;
// constexpr Real c_INVFOURPI = Real(1.0) / c_FOURPI;
// constexpr Real c_PIOVERTWO = Real(0.5) * c_PI;
// constexpr Real c_PIOVERFOUR = Real(0.25) * c_PI;

namespace fs = std::filesystem;

inline std::string to_lowercase(const std::string &s) {
    std::string out = s;
    std::transform(s.begin(), s.end(), out.begin(), ::tolower);
    return out;
}

template <typename T>
__host__ __device__ inline T infinity() {
    return std::numeric_limits<T>::infinity();
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