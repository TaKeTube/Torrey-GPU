#pragma once
#include "vector.h"

struct Ray {
    Vector3 origin;
    Vector3 dir;
    Real tmin;
    Real tmax;
};