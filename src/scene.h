#pragma once
#include <tuple>
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
    DeviceTexturePool textures;

    int num_lights;

    BVHNode* bvh_nodes;
    int bvh_root_id;
};

struct sceneInfo {
    Camera camera;
    int width, height;
    RenderOptions options;
    Vector3 background_color;
};

struct freeInfo {
    std::vector<DeviceTriangleMesh> device_meshes;
    std::vector<DeviceImage1> device_img1s;
    std::vector<DeviceImage3> device_img3s;
};

void build_bvh(Scene& scene);

inline std::tuple<deviceScene, freeInfo> device_scene_init(Scene& scene) {
    deviceScene dscene;
    Shape* deviceShapes;
    Light* deviceLights;
    Material* deviceMaterials;
    BVHNode* deviceBVHNodes;
    DeviceTriangleMesh* deviceTriangleMeshes;

    cudaMalloc((void **)&deviceShapes,    scene.shapes.size()    * sizeof(Shape));
    cudaMalloc((void **)&deviceLights,    scene.lights.size()    * sizeof(Light));
    cudaMalloc((void **)&deviceMaterials, scene.materials.size() * sizeof(Material));

    cudaMemcpy(deviceShapes,    scene.shapes.data(),    scene.shapes.size()    * sizeof(Shape),    cudaMemcpyHostToDevice);
    cudaMemcpy(deviceLights,    scene.lights.data(),    scene.lights.size()    * sizeof(Light),    cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaterials, scene.materials.data(), scene.materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    dscene.bvh_root_id = scene.bvh_root_id;
    dscene.shapes = deviceShapes;
    dscene.lights = deviceLights;
    dscene.materials = deviceMaterials;

    if(!scene.bvh_nodes.empty()){
        cudaMalloc((void **)&deviceBVHNodes,  scene.bvh_nodes.size() * sizeof(BVHNode));
        cudaMemcpy(deviceBVHNodes,  scene.bvh_nodes.data(), scene.bvh_nodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice);
        dscene.bvh_nodes = deviceBVHNodes;
    }else{
        dscene.bvh_nodes = nullptr;
    }

    dscene.num_lights = scene.lights.size();

    // Copy Meshes
    std::vector<DeviceTriangleMesh> device_meshes; 
    for(TriangleMesh mesh : scene.meshes) {
        device_meshes.push_back(device_mesh_init(mesh));
    }
    cudaMalloc((void **)&deviceTriangleMeshes, device_meshes.size() * sizeof(DeviceTriangleMesh));
    cudaMemcpy(deviceTriangleMeshes, device_meshes.data(), device_meshes.size() * sizeof(DeviceTriangleMesh), cudaMemcpyHostToDevice);
    dscene.meshes = deviceTriangleMeshes;

    // Copy Textures
    DeviceTexturePool& device_pool = dscene.textures;
    std::vector<DeviceImage1> device_img1s;
    std::vector<DeviceImage3> device_img3s;
    for(auto &img1 : scene.textures.image1s) {
        Real* deviceTexture1;
        cudaMalloc((void **)&deviceTexture1, img1.width * img1.height * sizeof(Real));
        cudaMemcpy(deviceTexture1, img1.data.data(), img1.width * img1.height * sizeof(Real), cudaMemcpyHostToDevice);
        device_img1s.push_back(DeviceImage1{img1.width, img1.height, deviceTexture1});
    }
    for(auto &img3 : scene.textures.image3s) {
        Vector3* deviceTexture3;
        cudaMalloc((void **)&deviceTexture3, img3.width * img3.height * sizeof(Vector3));
        cudaMemcpy(deviceTexture3, img3.data.data(), img3.width * img3.height * sizeof(Vector3), cudaMemcpyHostToDevice);
        device_img3s.push_back(DeviceImage3{img3.width, img3.height, deviceTexture3});
    }
    if(!scene.textures.image1s.empty()){
        cudaMalloc((void **)&device_pool.image1s, device_img1s.size() * sizeof(DeviceImage1));
        cudaMemcpy(device_pool.image1s, device_img1s.data(), device_img1s.size() * sizeof(DeviceImage1), cudaMemcpyHostToDevice);
    }
    if(!scene.textures.image3s.empty()){
        cudaMalloc((void **)&device_pool.image3s, device_img3s.size() * sizeof(DeviceImage3));
        cudaMemcpy(device_pool.image3s, device_img3s.data(), device_img3s.size() * sizeof(DeviceImage3), cudaMemcpyHostToDevice);
    }

    freeInfo free_info;
    free_info.device_meshes = std::move(device_meshes);
    free_info.device_img1s = std::move(device_img1s);
    free_info.device_img3s = std::move(device_img3s);
    return {dscene, free_info};
}

inline void device_scene_destruct(deviceScene& scene, freeInfo& free_info) {
    cudaFree(scene.shapes);
    cudaFree(scene.lights);
    cudaFree(scene.materials);
    cudaFree(scene.bvh_nodes);

    for(auto &m : free_info.device_meshes)
        device_mesh_destruct(m);
    cudaFree(scene.meshes);

    for(auto &img1 : free_info.device_img1s){
        cudaFree(img1.data);
    }
    cudaFree(scene.textures.image1s);

    for(auto &img3 : free_info.device_img3s){
        cudaFree(img3.data);
    }
    cudaFree(scene.textures.image3s);
}

__device__ inline std::optional<Intersection> bvh_intersect(const deviceScene& scene, const BVHNode &node, Ray ray) {
    // Recursive version. Which is not recommend in CUDA
    
    // if (node.primitive_id != -1) {
    //     return intersect_shape(scene, scene.shapes[node.primitive_id], ray);
    // }
    // const BVHNode &left = scene.bvh_nodes[node.left_node_id];
    // const BVHNode &right = scene.bvh_nodes[node.right_node_id];
    // std::optional<Intersection> isect_left;
    // if (intersect(left.box, ray)) {
    //     isect_left = bvh_intersect(scene, left, ray);
    //     if (isect_left) {
    //         ray.tmax = isect_left->t;
    //     }
    // }
    // if (intersect(right.box, ray)) {
    //     if (auto isect_right = bvh_intersect(scene, right, ray)) {
    //         return isect_right;
    //     }
    // }
    // return isect_left;

    // int node_ptr = 0;
    // std::optional<Intersection> intersection = {};
    // unsigned int bvhStack[64];
    // bvhStack[++node_ptr] = scene.bvh_root_id;

    // while(node_ptr)
    // {
    //     BVHNode curr_node = scene.bvh_nodes[bvhStack[node_ptr--]];

    //     if(!intersect(curr_node.box, ray))
    //         continue;
        
    //     if(curr_node.primitive_id != -1)
    //     {
    //         std::optional<Intersection> temp_intersection = intersect_shape(scene.meshes, scene.shapes[curr_node.primitive_id], ray);
    //         if (!intersection || (temp_intersection && temp_intersection->t < intersection->t)) 
    //             intersection = std::move(temp_intersection);
    //     }
    //     else
    //     {
    //         bvhStack[++node_ptr] = curr_node.right_node_id;
    //         bvhStack[++node_ptr] = curr_node.left_node_id;
    //     }
    // }
    // return intersection;

    int node_ptr = 0;
    bool is_left = true;
    std::optional<Intersection> intersection;
    intersection->t = infinity<Real>();
    unsigned int bvhStack[64];
    bvhStack[++node_ptr] = scene.bvh_root_id;

    while(node_ptr)
    {
        BVHNode curr_node = scene.bvh_nodes[bvhStack[node_ptr--]];

        if(!intersect(curr_node.box, ray))
            continue;
        
        if(curr_node.primitive_id != -1)
        {
            std::optional<Intersection> temp_intersection = intersect_shape(scene.meshes, scene.shapes[curr_node.primitive_id], ray);
            if(temp_intersection && temp_intersection->t < intersection->t){
                intersection = std::move(temp_intersection);
                if(is_left && intersection->t < ray.tmax) ray.tmax = intersection->t;
            }
            if(temp_intersection)
                is_left = false;
        }
        else
        {
            bvhStack[++node_ptr] = curr_node.right_node_id;
            bvhStack[++node_ptr] = curr_node.left_node_id;
            is_left = true;
        }
    }
    return infinity<Real>() == intersection->t ? std::nullopt : intersection;
}

__device__ inline std::optional<Intersection> scene_intersect(const deviceScene& scene, const Ray& r){
    if(nullptr != scene.bvh_nodes){
        return bvh_intersect(scene, scene.bvh_nodes[scene.bvh_root_id], r);
    }else{
        return {};
    }
}

__device__ inline bool scene_occluded(const deviceScene& scene, const Ray& r){
    if(nullptr != scene.bvh_nodes){
        std::optional<Intersection> v_ = bvh_intersect(scene, scene.bvh_nodes[scene.bvh_root_id], r);
        return v_ ? true : false;
    }else{
        return false;
    }
}

__device__ inline Vector3 trace_ray(const deviceScene& scene, const sceneInfo& scene_info, const Ray& ray, RNGr& rng){
    Ray r = ray;
    std::optional<Intersection> v_ = scene_intersect(scene, r);
    if(!v_) return scene_info.background_color;
    Intersection v = *v_;

    Vector3 radiance = {Real(0), Real(0), Real(0)};
    Vector3 throughput = {Real(1), Real(1), Real(1)};
    for(int i = 0; i <= scene_info.options.max_depth; ++i){
        if(v.area_light_id != -1) {
            const Light& light = scene.lights[v.area_light_id];
            if (auto* l = std::get_if<DiffuseAreaLight>(&light)){
                // std::cout << throughput << std::endl;
                radiance += throughput * l->intensity;
                break;
            }
        }

        
        Vector3 dir_in = -r.dir;
        const Material& m = scene.materials[v.material_id];
        bool is_specular = false;
        if(std::holds_alternative<Plastic>(m) || std::holds_alternative<Mirror>(m))
            is_specular = true;
        
        if(scene.num_lights > 0 && !is_specular && random_real(rng) <= 0.5){
            // Sampling Light
            int light_id = sample_light(scene.num_lights, rng);
            auto light = scene.lights[light_id];
            if (auto* l = std::get_if<DiffuseAreaLight>(&light)) {
                auto light_point = sample_on_light(scene.shapes, scene.meshes, *l, v.pos, rng);
                auto& [light_pos, light_n] = light_point;
                Real d = length(light_pos - v.pos);
                Vector3 light_dir = normalize(light_pos - v.pos);

                Real light_pdf = get_light_pdf(scene.lights, scene.shapes, scene.meshes, light_id, light_point, v.pos) * (d * d) / (fmax(dot(-light_n, light_dir), Real(0)) * scene.num_lights);
                if(light_pdf <= 0){
                    // std::cout << light_pdf << "light pdf break" << std::endl;
                    break;
                }
                Real bsdf_pdf = get_bsdf_pdf(m, dir_in, light_dir, v, scene.textures);
                if(bsdf_pdf <= 0){
                    // std::cout << "bsdf pdf break" << std::endl;
                    break;
                }
                
                SampleRecord record = {};
                record.dir_out = light_dir;
                Vector3 FG = eval(m, dir_in, record, v, scene.textures);

                r = Ray{v.pos, light_dir, c_EPSILON, infinity<Real>()};
                std::optional<Intersection> v_ = scene_intersect(scene, r);
                if(!v_){
                    // std::cout << "bg break" << std::endl;
                    radiance += throughput * scene_info.background_color;
                    break;
                }
                v = *v_;
                if(v.area_light_id == -1){
                    break;
                }
                throughput *= FG / (Real(0.5) * light_pdf + Real(0.5) * bsdf_pdf);
            }
        }else{
            // Sampling bsdf
            Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
            std::optional<SampleRecord> record_ = sample_bsdf(m, dir_in, v, scene.textures, rng);
            if(!record_){
                // std::cout << "record break" << std::endl;
                break;
            }
            SampleRecord& record = *record_;
            Vector3 FG = eval(m, dir_in, record, v, scene.textures);
            Vector3 dir_out = normalize(record.dir_out);
            Real bsdf_pdf = record.pdf;
            if(bsdf_pdf <= Real(0)){
                // std::cout << "pdf break" << std::endl;
                break;
            }
            r = Ray{v.pos, dir_out, c_EPSILON, infinity<Real>()};
            std::optional<Intersection> new_v_ = scene_intersect(scene, r);

            Real pdf = ((nullptr == scene.lights) || is_specular) ? bsdf_pdf : 0.5 * bsdf_pdf;

            if(!new_v_){
                // std::cout << "bg break" << std::endl;
                throughput *= FG / pdf;
                radiance += throughput * scene_info.background_color;
                break;
            }
            if(!is_specular && new_v_->area_light_id != -1){
                Vector3 &light_pos = new_v_->pos;
                Real d = length(light_pos - v.pos);
                Vector3 light_dir = normalize(light_pos - v.pos);
                Real light_pdf = get_light_pdf(scene.lights, scene.shapes, scene.meshes, new_v_->area_light_id, {new_v_->pos, new_v_->geo_normal}, v.pos) * (d * d) / (fmax(dot(-new_v_->geo_normal, light_dir), Real(0)) * scene.num_lights);
                if(light_pdf <= 0){
                    // std::cout << dot(-new_v_->geo_normal, light_dir) << std::endl;
                    break;
                }
                pdf += 0.5 * light_pdf;
            }
            throughput *= FG / pdf;
            v = *new_v_;
        }
    }
    return radiance;
}

__device__ inline void debug_log(deviceScene& scene, sceneInfo& scene_info) {
    
    printf("scene.bvh_nodes[0].left_node_id: %d\n", scene.bvh_nodes[0].left_node_id);
    printf("scene.bvh_root_id: %d\n", scene.bvh_root_id);

    Light &l = scene.lights[0];
    if (auto* ll = std::get_if<PointLight>(&l))
        printf("l[0] intensity.x: %f\n", ll->intensity.x);
    else if (auto* ll = std::get_if<DiffuseAreaLight>(&l))
        printf("l[0] intensity.x: %f\n", ll->intensity.x);

    Material &m = scene.materials[1];
    if(auto *mm = std::get_if<Diffuse>(&m))
        printf("m[0] color.x: %f\n", eval(mm->reflectance, Vector2{0, 0}, scene.textures).x);
    else if(auto *mm = std::get_if<Mirror>(&m))
        printf("m[0] color.x: %f\n", eval(mm->reflectance, Vector2{0, 0}, scene.textures).x);
    else if(auto *mm = std::get_if<Plastic>(&m))
        printf("m[0] color.x: %f\n", eval(mm->reflectance, Vector2{0, 0}, scene.textures).x);
    else if(auto *mm = std::get_if<Phong>(&m))
        printf("m[0] color.x: %f\n", eval(mm->reflectance, Vector2{0, 0}, scene.textures).x);
    else if(auto *mm = std::get_if<BlinnPhong>(&m))
        printf("m[0] color.x: %f\n", eval(mm->reflectance, Vector2{0, 0}, scene.textures).x);
    else if(auto *mm = std::get_if<BlinnPhongMicrofacet>(&m))
        printf("m[0] color.x: %f\n", eval(mm->reflectance, Vector2{0, 0}, scene.textures).x);
    else
        printf("unknown material\n");

    printf("scene.meshes[0].normals[0].xyz: %f %f %f\n", scene.meshes[0].normals[0].x, scene.meshes[0].normals[0].y, scene.meshes[0].normals[0].z);
    printf("scene.num_lights: %d\n", scene.num_lights);

    Shape &s = scene.shapes[0];
    if(auto *ss = std::get_if<Sphere>(&s))
        printf("shape[0] sphere->center.x: %f\n", ss->center.x);
    else if(auto *ss = std::get_if<Triangle>(&s))
        printf("shape[0] tri->face_index: %d\n", ss->face_index);
    else
        printf("unknown shape\n");

    printf("scene.textures.image3s[0].data[0].x: %f\n", scene.textures.image3s[0].data[0].x);

    printf("scene_info.background_color.x: %f\n", scene_info.background_color.x);
    printf("scene_info.camera.lookat.x: %f\n", scene_info.camera.lookat.x);
    printf("scene_info.height: %d\n", scene_info.height);
    printf("spp: %d\n", scene_info.options.spp);
}