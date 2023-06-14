#pragma once
#include "vector.cuh"

struct Ray
{
	Vector3 origin;
	Vector3 dir;
	double tmin;
	double tmax;
};