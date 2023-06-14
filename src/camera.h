#pragma once
#include "vector.h"
#include "parse_scene.h"

struct Camera {
    Vector3 lookfrom;
    Vector3 lookat;
    Vector3 up;
    Real vfov;
};

Camera from_parsed_camera(const ParsedCamera &pc);