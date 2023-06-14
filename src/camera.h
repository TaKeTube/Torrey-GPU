#pragma once
#include"vector.cuh"
#include"parse_scene.h"

struct Camera
{
	Vector3 lookfrom;
	Vector3 lookat;
	Vector3 ip;
	double vfov;
};

Camera from_parsed_camera(const ParsedCamera& pc);