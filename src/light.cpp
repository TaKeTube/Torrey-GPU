#include "light.h"
#include "scene.h"
#include <algorithm>

__device__ int sample_light(const deviceScene &scene, RNGf& rng) {
    return floor(random_double(rng) * scene.num_lights);
}

__device__ Real get_light_pdf(const deviceScene &scene, int light_id,
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

__device__ PointAndNormal sample_on_light_Point(const PointLight &l) {
    return {l.position, Vector3{0, 0, 0}};
}

__device__ PointAndNormal sample_on_light_DiffuseArea(const DiffuseAreaLight &l, const deviceScene& scene, const Vector3 &ref_pos, RNGf& rng)
{
    Shape s = scene.shapes[l.shape_id];
    return sample_on_shape(scene, s, ref_pos, rng);
}