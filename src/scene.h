#pragma once
#include "material.h"
#include "shape.h"
#include "light.h"
#include "bvh.h"
#include "camera.h"
#include "parse_scene.h"

struct RenderOptions {
    int spp = 4;
    int max_depth = -1;
};

struct Scene {
    Scene();
    Scene(const ParsedScene &scene);

    Camera camera;
    int width, height;
    std::vector<Shape> shapes;
    std::vector<TriangleMesh> meshes;
    std::vector<Light> lights;
    std::vector<Real> lights_power_pmf;
    std::vector<Real> lights_power_cdf;
    std::vector<Material> materials;
    TexturePool textures;
    Vector3 background_color;
    RenderOptions options;

    std::vector<BVHNode> bvh_nodes;
    int bvh_root_id;
};

std::optional<Intersection> bvh_intersect(const Scene &scene, const BVHNode &node, Ray ray);
std::optional<Intersection> scene_intersect(const Scene& scene, const Ray& r);
Real light_power(const Scene &scene, const Light &light);
bool scene_occluded(const Scene& scene, const Ray& r);
Vector3 trace_ray(const Scene& scene, const Ray& r, std::mt19937& rng);
Vector3 trace_ray_MIS(const Scene& scene, const Ray& ray, std::mt19937& rng);
Vector3 trace_ray_MIS_power(const Scene& scene, const Ray& ray, std::mt19937& rng);
void build_bvh(Scene& scene);