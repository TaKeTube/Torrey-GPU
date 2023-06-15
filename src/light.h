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

__device__ inline int sample_light(const deviceScene &scene, RNGf& rng) {
    return floor(random_double(rng) * scene.num_lights);
}

__device__ inline Real get_light_pdf(const deviceScene &scene, int light_id,
                   const PointAndNormal &light_point,
                   const Vector3 &ref_pos) {
    if(auto* l = std::get_if<DiffuseAreaLight>(&scene.lights[light_id])){
        auto shape = scene.shapes[l->shape_id];
        if(auto* s = std::get_if<Triangle>(&shape)){
            return 1/get_area(scene, *s);
        }else if(auto* s = std::get_if<Sphere>(&shape)){
            // return 1/get_area(*s);
            Real r = s->radius;
            Real d = length(light_point.position - ref_pos);
            return 1/(c_TWOPI * r * r * (1 - r / d));
        }
    }
    // std::cout << light_id << std::endl;
    return 0;
}

__device__ inline PointAndNormal sample_on_light_Point(const PointLight &l) {
    return {l.position, Vector3{0, 0, 0}};
}

__device__ inline PointAndNormal sample_on_light_DiffuseArea(const DiffuseAreaLight &l, const deviceScene& scene, const Vector3 &ref_pos, RNGf& rng)
{
    Shape s = scene.shapes[l.shape_id];
    return sample_on_shape(scene, s, ref_pos, rng);
}

__device__ inline PointAndNormal sample_on_light(const deviceScene &scene, const Light& l, const Vector3 &ref_pos, RNGf& rng) {
    if (auto* s = std::get_if<PointLight>(&l))
        return sample_on_light_Point(*s);
    else if (auto* s = std::get_if<DiffuseAreaLight>(&l))
        return sample_on_light_DiffuseArea(*s, scene, ref_pos, rng);
    else
        return PointAndNormal{ Vector3{0.0, 0.0, 0.0}, Vector3{0.0, 0.0, 0.0} };
}