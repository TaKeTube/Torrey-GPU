#pragma once
#include <variant>
#include <optional>
#include "vector.h"
#include "intersection.h"
#include "ray.h"

struct ShapeBase {
    int material_id = -1;
    int area_light_id = -1;
};

struct TriangleMesh : public ShapeBase {
    std::vector<Vector3> positions;
    std::vector<Vector3i> indices;
    std::vector<Vector3> normals;
    std::vector<Vector2> uvs;
};

struct Sphere : public ShapeBase {
    Vector3 center;
    Real radius;
};

struct Triangle {
    int face_index;
    const TriangleMesh *mesh;
    int area_light_id = -1;
};

Vector2 get_sphere_uv(const Vector3& p);
Vector2 get_triangle_uv();

using Shape = std::variant<Sphere, Triangle>;

struct intersect_op {
    std::optional<Intersection> operator()(const Sphere &s) const;
    std::optional<Intersection> operator()(const Triangle &s) const;

    const Ray& r;
};

struct sample_on_shape_op {
    PointAndNormal operator()(const Sphere &s) const;
    PointAndNormal operator()(const Triangle &s) const;

    const Vector3 &ref_pos;
    std::mt19937& rng;
};

inline PointAndNormal sample_on_shape(const Shape& shape, const Vector3 &ref_pos, std::mt19937& rng) {
    return std::visit(sample_on_shape_op{ref_pos, rng}, shape);
}

struct get_area_op {
    Real operator()(const Sphere &s) const;
    Real operator()(const Triangle &s) const;
};

inline Real get_area(const Shape& shape) {
    return std::visit(get_area_op{}, shape);
}