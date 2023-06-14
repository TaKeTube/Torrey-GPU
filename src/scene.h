#pragma once
#include "material.h"
#include "shape.h"
#include "light.h"
#include "bvh.h"
#include "camera.h"
#include "parse_scene.h"
#include "rng.h"

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
    std::vector<Material> materials;
    TexturePool textures;
    Vector3 background_color;
    RenderOptions options;

    std::vector<BVHNode> bvh_nodes;
    int bvh_root_id;
};

struct deviceScene {
    Shape* shapes;
    DeviceTriangleMesh* meshes;
    Light* lights;
    Material* materials;
    TexturePool textures;

    int num_lights;
    int num_meshes;

    BVHNode* bvh_nodes;
    int bvh_root_id;
};

struct sceneInfo {
    Camera camera;
    int width, height;
    RenderOptions options;
    Vector3 background_color;
};

__device__ std::optional<Intersection> bvh_intersect(const deviceScene &scene, const BVHNode &node, Ray ray);
__device__ std::optional<Intersection> scene_intersect(const deviceScene& scene, const Ray& r);
__device__ bool scene_occluded(const deviceScene& scene, const Ray& r);
__device__ Vector3 trace_ray(const deviceScene& scene, const sceneInfo& scene_info,const Ray& ray, RNGf& rng);
void build_bvh(Scene& scene);

inline deviceScene device_scene_init(Scene& scene) {
    deviceScene dscene;
    Shape* deviceShapes;
    Light* deviceLights;
    Material* deviceMaterials;
    BVHNode* deviceBVHNodes;
    DeviceTriangleMesh* deviceTriangleMeshes;

    cudaMalloc((void **)&deviceShapes,    scene.shapes.size()    * sizeof(Shape));
    cudaMalloc((void **)&deviceLights,    scene.lights.size()    * sizeof(Light));
    cudaMalloc((void **)&deviceMaterials, scene.shapes.size()    * sizeof(Material));

    cudaMemcpy(deviceShapes,    scene.shapes.data(),    scene.shapes.size()    * sizeof(Shape),    cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLights,    scene.lights.data(),    scene.lights.size()    * sizeof(Light),    cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaterials, scene.shapes.data(),    scene.shapes.size()    * sizeof(Material), cudaMemcpyHostToDevice);

    dscene.bvh_root_id = scene.bvh_root_id;
    dscene.shapes = deviceShapes;
    dscene.lights = deviceLights;  
    dscene.materials = deviceMaterials;

    if(!scene.bvh_nodes.empty()){
        cudaMalloc((void **)&deviceBVHNodes,  scene.bvh_nodes.size() * sizeof(BVHNode));
        cudaMemcpy(deviceBVHNodes,  scene.bvh_nodes.data(), scene.bvh_nodes.size() * sizeof(BVHNode),  cudaMemcpyHostToDevice);
        dscene.bvh_nodes = deviceBVHNodes;
    }else{
        dscene.bvh_nodes = nullptr;
    }

    dscene.num_lights = scene.lights.size();

    std::vector<DeviceTriangleMesh> device_meshes; 
    for(TriangleMesh mesh : scene.meshes) {
        device_meshes.push_back(device_mesh_init(mesh));
    }
    cudaMalloc((void **)&deviceTriangleMeshes, device_meshes.size() * sizeof(DeviceTriangleMesh));
    cudaMemcpy(deviceTriangleMeshes, device_meshes.data(), device_meshes.size() * sizeof(DeviceTriangleMesh), cudaMemcpyHostToDevice);
    dscene.meshes = deviceTriangleMeshes;

    return dscene;
}

inline deviceScene device_scene_destruct(deviceScene& scene) {
    cudaFree(scene.shapes);
    cudaFree(scene.lights);
    cudaFree(scene.materials);
    cudaFree(scene.bvh_nodes);
    for(int i = 0; i < scene.num_meshes; ++i)
        device_mesh_destruct(scene.meshes[i]);
}