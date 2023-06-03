#pragma once
#include "vector.h"

struct Intersection {
    Vector3 pos;
    Vector3 geo_normal;
    Vector3 shading_normal;
    Vector2 uv;
    Real t;
    int material_id;
    int area_light_id;
};

struct PointAndNormal {
	Vector3 position;
	Vector3 normal;
};