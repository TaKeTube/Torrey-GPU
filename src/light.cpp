#include "light.h"
#include "scene.h"
#include <algorithm>

int sample_light(const Scene &scene, std::mt19937& rng) {
    return floor(random_double(rng) * scene.lights.size());
}

int sample_light_power(const Scene &scene, std::mt19937& rng) {
    const std::vector<Real> &power_cdf = scene.lights_power_cdf;
    Real u = random_double(rng);
    int size = power_cdf.size() - 1;
    assert(size > 0);
    const Real *ptr = std::upper_bound(power_cdf.data(), power_cdf.data() + size + 1, u);
    int offset = std::clamp(int(ptr - power_cdf.data() - 1), 0, size - 1);
    return offset;
}

Real get_light_pmf(const Scene &scene, int id) {
    const std::vector<Real> &pmf = scene.lights_power_pmf;
    assert(id >= 0 && id < (int)pmf.size());
    return pmf[id];
}

Real light_power(const Scene &scene, const Light &light) {
    if(auto* l = std::get_if<DiffuseAreaLight>(&light)){
        return luminance(l->intensity) * get_area(scene.shapes[l->shape_id]) * c_PI;
    }
    return 0;
}

Real get_light_pdf(const Scene &scene, int light_id,
                   const PointAndNormal &light_point,
                   const Vector3 &ref_pos) {
    if(auto* l = std::get_if<DiffuseAreaLight>(&scene.lights[light_id])){
        auto shape = scene.shapes[l->shape_id];
        if(auto* s = std::get_if<Triangle>(&shape)){
            return 1/get_area(*s);
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

PointAndNormal sample_on_light_op::operator()(const PointLight &l) const {
    return {l.position, Vector3{0, 0, 0}};
}

PointAndNormal sample_on_light_op::operator()(const DiffuseAreaLight &l) const {
    return std::visit(sample_on_shape_op{ref_pos, rng}, scene.shapes.at(l.shape_id));
}