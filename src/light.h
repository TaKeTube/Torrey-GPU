#pragma once
#include <variant>
#include "vector.h"
#include "intersection.h"
#include "shape.h"

struct Scene;

struct PointLight {
    Vector3 intensity;
    Vector3 position;    
};

struct DiffuseAreaLight {
    int shape_id;
    Vector3 intensity;
};

using Light = std::variant<PointLight, DiffuseAreaLight>;

__device__ int sample_light(const Scene &scene, std::mt19937& rng);
__device__ int sample_light_power(const Scene &scene, std::mt19937& rng);
__device__ Real get_light_pmf(const Scene &scene, int id);
__device__ Real get_light_pdf(const Scene &scene, int light_id,
                   const PointAndNormal &light_point,
                   const Vector3 &ref_pos
);

__device__ PointAndNormal sample_on_light_Point(const PointLight &l);
__device__ PointAndNormal sample_on_light_DiffuseArea(const DiffuseAreaLight &l, const Scene& scene, const Vector3 &ref_pos, std::mt19937& rng);

__device__ inline PointAndNormal sample_on_light(const Scene &scene, const Light& l, const Vector3 &ref_pos, std::mt19937& rng) {
    if(auto *s = std::get_if<PointLight>(&l))
        return sample_on_light_Point(*s);
    else if(auto *s = std::get_if<DiffuseAreaLight>(&l))
        return sample_on_light_DiffuseArea(*s, scene, ref_pos, rng);
}


