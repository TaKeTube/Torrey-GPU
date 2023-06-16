#pragma once
#include <variant>
#include "vector.h"
#include "intersection.h"
#include "shape.h"
#include "rng.h"

struct PointLight {
    Vector3 intensity;
    Vector3 position;    
};

struct DiffuseAreaLight {
    int shape_id;
    Vector3 intensity;
};

using Light = std::variant<PointLight, DiffuseAreaLight>;
using LightArr = Light*;

__device__ inline int sample_light(int num_lights, RNGr& rng) {
    return floor(random_real(rng) * num_lights);
}

__device__ inline Real get_light_pdf(const LightArr &lights, const ShapeArr &shapes, const TriangleMeshArr &meshes, int light_id,
                   const PointAndNormal &light_point,
                   const Vector3 &ref_pos) {
    if(auto* l = std::get_if<DiffuseAreaLight>(&lights[light_id])){
        auto shape = shapes[l->shape_id];
        if(auto* s = std::get_if<Triangle>(&shape)){
            return 1/get_area(meshes, *s);
        }else if(auto* s = std::get_if<Sphere>(&shape)){
            // return 1/get_area(*s);
            Real r = s->radius;
            Real d = length(light_point.position - ref_pos);
            return 1/(c_TWOPI * r * r * (1 - r / d));
        }
    }
    return 0;
}

__device__ inline PointAndNormal sample_on_light_Point(const PointLight &l) {
    return {l.position, Vector3{0, 0, 0}};
}

__device__ inline PointAndNormal sample_on_light_DiffuseArea(const DiffuseAreaLight &l, const ShapeArr& shapes, const TriangleMeshArr &meshes, const Vector3 &ref_pos, RNGr& rng)
{
    Shape s = shapes[l.shape_id];
    return sample_on_shape(meshes, s, ref_pos, rng);
}

__device__ inline PointAndNormal sample_on_light(const ShapeArr &shapes, const TriangleMeshArr &meshes, const Light& l, const Vector3 &ref_pos, RNGr& rng) {
    if (auto* s = std::get_if<PointLight>(&l))
        return sample_on_light_Point(*s);
    else if (auto* s = std::get_if<DiffuseAreaLight>(&l))
        return sample_on_light_DiffuseArea(*s, shapes, meshes, ref_pos, rng);
    else
        return PointAndNormal{ Vector3{0.0, 0.0, 0.0}, Vector3{0.0, 0.0, 0.0} };
}