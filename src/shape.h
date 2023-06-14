#pragma once
#include <variant>
#include <optional>
#include "vector.h"
#include "intersection.h"
#include "ray.h"

struct deviceScene;

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

struct DeviceTriangleMesh : public ShapeBase {
    Vector3* positions;
    Vector3i* indices;
    Vector3* normals;
    Vector2* uvs;
};

struct Sphere : public ShapeBase {
    Vector3 center;
    Real radius;
};

struct Triangle {
    int face_index;
    int mesh_id;
    int area_light_id = -1;
};

using Shape = std::variant<Sphere, Triangle>;

__device__ Vector2 get_sphere_uv(const Vector3& p);
__device__ Vector2 get_triangle_uv();

__device__ std::optional<Intersection> intersect_triangle(const Triangle& tri, const Ray& r, const deviceScene& scene);
__device__ std::optional<Intersection> intersect_sphere(const Sphere& s, const Ray& r);
__device__ PointAndNormal sample_on_shape_triangle(const Triangle &t, const Vector3 &ref_pos, RNGf& rng, const deviceScene& scene);
__device__ PointAndNormal sample_on_shape_sphere(const Sphere &s, const Vector3 &ref_pos, RNGf& rng);
__device__ Real get_area_sphere(const Sphere &t);
__device__ Real get_area_triangle(const Triangle &t, const deviceScene& scene);

__device__ inline std::optional<Intersection> intersect_shape(const deviceScene& scene, const Shape& shape, const Ray& r){
    if(auto *s = std::get_if<Sphere>(&shape))
        return intersect_sphere(*s, r);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return intersect_triangle(*s, r, scene);
    else
        return {};  
}

__device__ inline PointAndNormal sample_on_shape(const deviceScene& scene, const Shape& shape, const Vector3 &ref_pos, RNGf& rng) {
    if(auto *s = std::get_if<Sphere>(&shape))
        return sample_on_shape_sphere(*s, ref_pos, rng);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return sample_on_shape_triangle(*s, ref_pos, rng, scene);
    else
        return {};
}

__device__ inline Real get_area(const deviceScene& scene, const Shape& shape) {
    if(auto *s = std::get_if<Sphere>(&shape))
        return get_area_sphere(*s);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return get_area_triangle(*s, scene);
    else
        return 0;
}

inline DeviceTriangleMesh device_mesh_init(TriangleMesh &mesh) {
    DeviceTriangleMesh device_mesh;
    Vector3* devicePositions;
    Vector3i* deviceIndices;
    Vector3* deviceMormals;
    Vector2* deviceUVs;

    cudaMalloc((void **)&devicePositions, mesh.positions.size() * sizeof(Vector3));
    cudaMalloc((void **)&deviceIndices,   mesh.indices.size()   * sizeof(Vector3i));

    cudaMemcpy(devicePositions, mesh.positions.data(), mesh.positions.size() * sizeof(Vector3),  cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndices,   mesh.indices.data(),   mesh.indices.size()   * sizeof(Vector3i), cudaMemcpyHostToDevice);

    device_mesh.positions = devicePositions;
    device_mesh.indices = deviceIndices;
    
    if(!mesh.normals.empty()){
        cudaMalloc((void **)&deviceMormals,   mesh.normals.size()   * sizeof(Vector3));
        cudaMemcpy(deviceMormals,   mesh.normals.data(),   mesh.normals.size()   * sizeof(Vector3),  cudaMemcpyHostToDevice);
        device_mesh.normals = deviceMormals;
    }else{
        device_mesh.normals = nullptr;
    }

    if(!mesh.uvs.empty()){
        cudaMalloc((void **)&deviceUVs,       mesh.uvs.size()       * sizeof(Vector2));
        cudaMemcpy(deviceUVs,       mesh.uvs.data(),       mesh.uvs.size()       * sizeof(Vector2),  cudaMemcpyHostToDevice);
        device_mesh.uvs = deviceUVs;
    }else{
        device_mesh.uvs = nullptr;
    }

    return device_mesh;
}

inline void device_mesh_destruct(DeviceTriangleMesh &m) {
    cudaFree(m.positions);
    cudaFree(m.indices);
    cudaFree(m.normals);
    cudaFree(m.uvs);
}