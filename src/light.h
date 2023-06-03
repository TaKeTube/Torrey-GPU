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

struct sample_on_light_op {
    PointAndNormal operator()(const PointLight &l) const;
    PointAndNormal operator()(const DiffuseAreaLight &l) const;

    const Scene &scene;
    const Vector3 &ref_pos;
    std::mt19937& rng;
};

int sample_light(const Scene &scene, std::mt19937& rng);
int sample_light_power(const Scene &scene, std::mt19937& rng);
Real get_light_pmf(const Scene &scene, int id);
Real get_light_pdf(const Scene &scene, int light_id,
                   const PointAndNormal &light_point,
                   const Vector3 &ref_pos
);

inline PointAndNormal sample_on_light(const Scene &scene, const Light& l, const Vector3 &ref_pos, std::mt19937& rng) {
    return std::visit(sample_on_light_op{scene, ref_pos, rng}, l);
}