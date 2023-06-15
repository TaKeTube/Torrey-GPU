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
using ShapeArr = Shape*;
using TriangleMeshArr = DeviceTriangleMesh*;

__device__ inline Vector2 get_sphere_uv(const Vector3& p) {
    // p: a given point on the sphere of radius one, centered at the origin.
    Real theta = acos(-p.y);
    Real phi = atan2(-p.z, p.x) + c_PI;

    Real u = phi / (2*c_PI);
    Real v = - theta / c_PI;
    return {u, v};
}

__device__ inline std::optional<Intersection> intersect_sphere(const Sphere& s, const Ray& r) {
    Vector3 oc = r.origin - s.center;
    Real a = dot(r.dir, r.dir);
    Real half_b = dot(oc, r.dir);
    Real c = dot(oc, oc) - s.radius*s.radius;

    Real discriminant = half_b*half_b - a*c;
    if (discriminant < 0) 
        return {};
    Real sqrtd = sqrt(discriminant);

    Real root = (-half_b - sqrtd) / a;
    if (root < r.tmin || r.tmax < root) {
        root = (-half_b + sqrtd) / a;
        if (root < r.tmin || r.tmax < root)
            return {};
    }

    Intersection v;
    v.t = root;
    v.pos = r.origin + r.dir * v.t;
    v.geo_normal = normalize(v.pos - s.center);
    v.geo_normal = dot(r.dir, v.geo_normal) < 0 ? v.geo_normal : -v.geo_normal;
    v.shading_normal = v.geo_normal;
    v.material_id = s.material_id;
    v.uv = get_sphere_uv(v.geo_normal);
    v.area_light_id = s.area_light_id;

    return v;
}

__device__ inline std::optional<Intersection> intersect_triangle(const Triangle& tri, const Ray& r, const TriangleMeshArr &meshes) {
    const DeviceTriangleMesh &mesh = meshes[tri.mesh_id];
    const Vector3i &indices = mesh.indices[tri.face_index];

    Vector3 v0 = mesh.positions[indices.x];
    Vector3 v1 = mesh.positions[indices.y];
    Vector3 v2 = mesh.positions[indices.z];
    Vector3 e1, e2, h, s, q;
    Real a, f, u, v;
    e1 = v1 - v0;
    e2 = v2 - v0;
    h = cross(r.dir, e2);
    a = dot(e1, h);

    if (a > -c_EPSILON && a < c_EPSILON)
        return {};    // This ray is parallel to this triangle.

    f = 1.0 / a;
    s = r.origin - v0;
    u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
        return {};

    q = cross(s, e1);
    v = f * dot(r.dir, q);

    if (v < 0.0 || u + v > 1.0)
        return {};

    // At this stage we can compute t to find out where the intersection point is on the line.
    Real t = f * dot(e2, q);

    if (t < r.tmin || r.tmax < t) // ray intersection
        return {};
    else {
        Intersection inter;
        inter.t = t;
        inter.pos = r.origin + r.dir * t;
        inter.geo_normal = normalize(cross(e1, e2));
        inter.geo_normal = dot(r.dir, inter.geo_normal) < 0 ? inter.geo_normal : -inter.geo_normal;
        inter.material_id = mesh.material_id;
        inter.area_light_id = tri.area_light_id;
        // Compute uv
        if (nullptr == mesh.uvs) {
            inter.uv = Vector2(u, v);
        } else {
            Vector2 uv0 = mesh.uvs[indices.x];
            Vector2 uv1 = mesh.uvs[indices.y];
            Vector2 uv2 = mesh.uvs[indices.z];
            inter.uv = (1 - u - v) * uv0 + u * uv1 + v * uv2;
            // if(inter.uv.y == 1 || inter.uv.y == 0){
            //     std::cout << uv0 << uv1 << uv2 << std::endl;
            // }
        }
        // Compute shading normal
        if (nullptr == mesh.normals) {
            inter.shading_normal = inter.geo_normal;
        } else {
            Vector3 n0 = mesh.normals[indices.x];
            Vector3 n1 = mesh.normals[indices.y];
            Vector3 n2 = mesh.normals[indices.z];
            inter.shading_normal = normalize((1 - u - v) * n0 + u * n1 + v * n2);
        }
        return inter;
    }
}

__device__ inline PointAndNormal sample_on_shape_sphere(const Sphere &s, const Vector3 &ref_pos, RNGf& rng) {
    Real u1 = random_double(rng);
    Real u2 = random_double(rng);

    Real r = s.radius;
    Real d = length(s.center - ref_pos);
    Real z = 1 + u1 * (r / d - 1);
    Real z2 = z * z;
    Real sin_theta = std::sqrt(std::clamp(1 - z2, Real(0), Real(1)));
    
    Vector3 local_p = normalize(Vector3{
        cos(2 * c_PI * u2) * sin_theta,
        sin(2 * c_PI * u2) * sin_theta,
        z
    });

    Vector3 normal = normalize(to_world(normalize(ref_pos - s.center), local_p));
    Vector3 point = s.center + s.radius * normal;
    return {point, normal};
}

__device__ inline PointAndNormal sample_on_shape_triangle(const Triangle &t, const Vector3 &ref_pos, RNGf& rng, const TriangleMeshArr &meshes) {
    const DeviceTriangleMesh &mesh = meshes[t.mesh_id];
    const Vector3i &indices = mesh.indices[t.face_index];

    Vector3 v0 = mesh.positions[indices.x];
    Vector3 v1 = mesh.positions[indices.y];
    Vector3 v2 = mesh.positions[indices.z];

    Real u1 = random_double(rng);
    Real u2 = random_double(rng);

    Real b1 = 1 - sqrt(u1);
    Real b2 = sqrt(u1) * u2;

    Vector3 point = (1 - b1 - b2) * v0 + b1 * v1 + b2 * v2;
    Vector3 normal = normalize(cross(v1 - v0, v2 - v0));

    Vector3 shading_normal;
    if (nullptr == mesh.normals) {
        shading_normal = normal;
    } else {
        Vector3 n0 = mesh.normals[indices.x];
        Vector3 n1 = mesh.normals[indices.y];
        Vector3 n2 = mesh.normals[indices.z];
        shading_normal = (1 - b1 - b2) * n0 + b1 * n1 + b2 * n2;
    }

    return {point, dot(shading_normal, normal) > 0 ? normal : -normal};
}

__device__ inline Real get_area_sphere(const Sphere &s) {
    return 4*c_PI*s.radius*s.radius;
}

__device__ inline Real get_area_triangle(const Triangle &t, const TriangleMeshArr &meshes) {
    const DeviceTriangleMesh &mesh = meshes[t.mesh_id];
    const Vector3i &indices = mesh.indices[t.face_index];

    Vector3 v0 = mesh.positions[indices.x];
    Vector3 v1 = mesh.positions[indices.y];
    Vector3 v2 = mesh.positions[indices.z];

    return length(cross(v1 - v0, v2 - v0)) / 2;
}

__device__ inline std::optional<Intersection> intersect_shape(const TriangleMeshArr &meshes, const Shape& shape, const Ray& r){
    if(auto *s = std::get_if<Sphere>(&shape))
        return intersect_sphere(*s, r);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return intersect_triangle(*s, r, meshes);
    else
        return {};
}

__device__ inline PointAndNormal sample_on_shape(const TriangleMeshArr &meshes, const Shape& shape, const Vector3 &ref_pos, RNGf& rng) {
    if(auto *s = std::get_if<Sphere>(&shape))
        return sample_on_shape_sphere(*s, ref_pos, rng);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return sample_on_shape_triangle(*s, ref_pos, rng, meshes);
    else
        return {};
}

__device__ inline Real get_area(const TriangleMeshArr &meshes, const Shape& shape) {
    if(auto *s = std::get_if<Sphere>(&shape))
        return get_area_sphere(*s);
    else if(auto *s = std::get_if<Triangle>(&shape))
        return get_area_triangle(*s, meshes);
    else
        return 0;
}

inline DeviceTriangleMesh device_mesh_init(TriangleMesh &mesh) {
    DeviceTriangleMesh device_mesh;
    Vector3* devicePositions;
    Vector3i* deviceIndices;
    Vector3* deviceNormals;
    Vector2* deviceUVs;

    device_mesh.area_light_id = mesh.area_light_id;
    device_mesh.material_id = mesh.material_id;

    cudaMalloc((void **)&devicePositions, mesh.positions.size() * sizeof(Vector3));
    cudaMalloc((void **)&deviceIndices,   mesh.indices.size()   * sizeof(Vector3i));

    cudaMemcpy(devicePositions, mesh.positions.data(), mesh.positions.size() * sizeof(Vector3),  cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIndices,   mesh.indices.data(),   mesh.indices.size()   * sizeof(Vector3i), cudaMemcpyHostToDevice);

    device_mesh.positions = devicePositions;
    device_mesh.indices = deviceIndices;
    
    if(!mesh.normals.empty()){
        cudaMalloc((void **)&deviceNormals, mesh.normals.size() * sizeof(Vector3));
        cudaMemcpy(deviceNormals, mesh.normals.data(), mesh.normals.size() * sizeof(Vector3), cudaMemcpyHostToDevice);
        device_mesh.normals = deviceNormals;
    }else{
        device_mesh.normals = nullptr;
    }

    if(!mesh.uvs.empty()){
        cudaMalloc((void **)&deviceUVs, mesh.uvs.size() * sizeof(Vector2));
        cudaMemcpy(deviceUVs, mesh.uvs.data(), mesh.uvs.size() * sizeof(Vector2), cudaMemcpyHostToDevice);
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