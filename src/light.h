#pragma once
#include <variant>
#include "vector.h"
#include "intersection.h"
#include "shape.h"

struct deviceScene;

struct PointLight {
    Vector3 intensity;
    Vector3 position;    
};

struct DiffuseAreaLight {
    int shape_id;
    Vector3 intensity;
};

using Light = std::variant<PointLight, DiffuseAreaLight>;

__device__ int sample_light(const deviceScene &scene, RNGf& rng);
__device__ Real get_light_pdf(const deviceScene &scene, int light_id,
                   const PointAndNormal &light_point,
                   const Vector3 &ref_pos
);

__device__ PointAndNormal sample_on_light_Point(const PointLight &l);
__device__ PointAndNormal sample_on_light_DiffuseArea(const DiffuseAreaLight &l, const deviceScene& scene, const Vector3 &ref_pos, RNGf& rng);

__device__ inline PointAndNormal sample_on_light(const deviceScene &scene, const Light& l, const Vector3 &ref_pos, RNGf& rng) {
    if (auto* s = std::get_if<PointLight>(&l))
        return sample_on_light_Point(*s);
    else if (auto* s = std::get_if<DiffuseAreaLight>(&l))
        return sample_on_light_DiffuseArea(*s, scene, ref_pos, rng);
    else
        return PointAndNormal{ Vector3{0.0, 0.0, 0.0}, Vector3{0.0, 0.0, 0.0} };
}


