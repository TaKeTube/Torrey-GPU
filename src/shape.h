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

using Shape = std::variant<Sphere, Triangle>;

__device__ Vector2 get_sphere_uv(const Vector3& p);
__device__ Vector2 get_triangle_uv();

__device__ std::optional<Intersection> intersect_triangle(const Triangle& tri, const Ray& r);
__device__ std::optional<Intersection> intersect_sphere(const Sphere& s, const Ray& r);
__device__ PointAndNormal sample_on_shape_triangle(const Triangle &t, const Vector3 &ref_pos, std::mt19937& rng);
__device__ PointAndNormal sample_on_shape_sphere(const Sphere &s, const Vector3 &ref_pos, std::mt19937& rng);
__device__ Real get_area_sphere(const Sphere &t);
__device__ Real get_area_triangle(const Triangle &t);

__device__ inline std::optional<Intersection> intersect_shape(const Shape& shape, const Ray& r){
    if(auto *s = std::get_if<Sphere>(&shape))
        return intersect_sphere(*s, r);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return intersect_triangle(*s, r);
    else
        return {};  
}

__device__ inline PointAndNormal sample_on_shape(const Shape& shape, const Vector3 &ref_pos, std::mt19937& rng) {
    if(auto *s = std::get_if<Sphere>(&shape))
        return sample_on_shape_sphere(*s, ref_pos, rng);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return sample_on_shape_triangle(*s, ref_pos, rng);
    else
        return {};
}

__device__ inline Real get_area(const Shape& shape) {
    if(auto *s = std::get_if<Sphere>(&shape))
        return get_area_sphere(*s);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return get_area_triangle(*s);
    else
        return 0;
}