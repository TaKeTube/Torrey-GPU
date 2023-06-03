#include "hw4.h"
#include "parse_scene.h"
#include "scene.h"
#include "timer.h"
#include "parallel.h"
#include "progressreporter.h"

Image3 hw_4_1(const std::vector<std::string> &params) {
    // Homework 4.1: diffuse interreflection
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    int max_depth = 50;
    std::string filename;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-max_depth") {
            max_depth = std::stoi(params[++i]);
        } else if (filename.empty()) {
            filename = params[i];
        }
    }

    Timer timer;
    std::cout << "Parsing and constructing scene " << params[0] << "." << std::endl;
    tick(timer);
    ParsedScene pscene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;
    UNUSED(pscene);

    Scene scene(pscene);
    scene.options.max_depth = max_depth;

    Image3 img(scene.width, scene.height);

    Camera &cam = scene.camera;

    Real theta = cam.vfov / 180 * c_PI;
    Real h = tan(theta/2);
    Real viewport_height = 2.0 * h;
    Real viewport_width = viewport_height / img.height * img.width;

    Vector3 w = normalize(cam.lookfrom - cam.lookat);
    Vector3 u = normalize(cross(cam.up, w));
    Vector3 v = cross(w, u);

    // Build BVH
    std::cout << "Building BVH..." << std::endl;
    tick(timer);
    build_bvh(scene);
    std::cout << "Finish building BVH. Took " << tick(timer) << " seconds." << std::endl;

    constexpr int tile_size = 16;
    int num_tiles_x = (img.width + tile_size - 1) / tile_size;
    int num_tiles_y = (img.height + tile_size - 1) / tile_size;
    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    
    std::cout << "Rendering..." << std::endl;
    tick(timer);
    parallel_for([&](const Vector2i &tile) {
        std::mt19937 rng{std::random_device{}()};
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, img.width);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, img.height);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Vector3 color = {0, 0, 0};
                for (int i = 0; i < scene.options.spp; i++){
                    Ray r = {cam.lookfrom,
                            normalize(
                            u * ((x + random_double(rng)) / img.width - Real(0.5)) * viewport_width +
                            v * ((y + random_double(rng)) / img.height - Real(0.5)) * viewport_height -
                            w),
                            c_EPSILON,
                            infinity<Real>()};
                    color += trace_ray(scene, r, rng);
                }
                img(x, img.height - y - 1) = color / Real(scene.options.spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    std::cout << std::endl << "Finish building rendering. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

Image3 hw_4_2(const std::vector<std::string> &params) {
    // Homework 4.2: adding more materials
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    int max_depth = 50;
    std::string filename;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-max_depth") {
            max_depth = std::stoi(params[++i]);
        } else if (filename.empty()) {
            filename = params[i];
        }
    }

    Timer timer;
    std::cout << "Parsing and constructing scene " << params[0] << "." << std::endl;
    tick(timer);
    ParsedScene pscene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;
    UNUSED(pscene);

    Scene scene(pscene);
    scene.options.max_depth = max_depth;

    Image3 img(scene.width, scene.height);

    Camera &cam = scene.camera;

    Real theta = cam.vfov / 180 * c_PI;
    Real h = tan(theta/2);
    Real viewport_height = 2.0 * h;
    Real viewport_width = viewport_height / img.height * img.width;

    Vector3 w = normalize(cam.lookfrom - cam.lookat);
    Vector3 u = normalize(cross(cam.up, w));
    Vector3 v = cross(w, u);

    // Build BVH
    std::cout << "Building BVH..." << std::endl;
    tick(timer);
    build_bvh(scene);
    std::cout << "Finish building BVH. Took " << tick(timer) << " seconds." << std::endl;

    constexpr int tile_size = 16;
    int num_tiles_x = (img.width + tile_size - 1) / tile_size;
    int num_tiles_y = (img.height + tile_size - 1) / tile_size;
    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    
    std::cout << "Rendering..." << std::endl;
    tick(timer);
    parallel_for([&](const Vector2i &tile) {
        std::mt19937 rng{std::random_device{}()};
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, img.width);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, img.height);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Vector3 color = {0, 0, 0};
                for (int i = 0; i < scene.options.spp; i++){
                    Ray r = {cam.lookfrom,
                            normalize(
                            u * ((x + random_double(rng)) / img.width - Real(0.5)) * viewport_width +
                            v * ((y + random_double(rng)) / img.height - Real(0.5)) * viewport_height -
                            w),
                            c_EPSILON,
                            infinity<Real>()};
                    color += trace_ray(scene, r, rng);
                }
                img(x, img.height - y - 1) = color / Real(scene.options.spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    std::cout << std::endl << "Finish building rendering. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

Image3 hw_4_3(const std::vector<std::string> &params) {
    // Homework 4.3: multiple importance sampling
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    int max_depth = 50;
    std::string filename;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-max_depth") {
            max_depth = std::stoi(params[++i]);
        } else if (filename.empty()) {
            filename = params[i];
        }
    }

    Timer timer;
    std::cout << "Parsing and constructing scene " << params[0] << "." << std::endl;
    tick(timer);
    ParsedScene pscene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;
    UNUSED(pscene);

    Scene scene(pscene);
    scene.options.max_depth = max_depth;

    Image3 img(scene.width, scene.height);

    Camera &cam = scene.camera;

    Real theta = cam.vfov / 180 * c_PI;
    Real h = tan(theta/2);
    Real viewport_height = 2.0 * h;
    Real viewport_width = viewport_height / img.height * img.width;

    Vector3 w = normalize(cam.lookfrom - cam.lookat);
    Vector3 u = normalize(cross(cam.up, w));
    Vector3 v = cross(w, u);

    // Build BVH
    std::cout << "Building BVH..." << std::endl;
    tick(timer);
    build_bvh(scene);
    std::cout << "Finish building BVH. Took " << tick(timer) << " seconds." << std::endl;

    constexpr int tile_size = 16;
    int num_tiles_x = (img.width + tile_size - 1) / tile_size;
    int num_tiles_y = (img.height + tile_size - 1) / tile_size;
    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    
    std::cout << "Rendering..." << std::endl;
    tick(timer);
    parallel_for([&](const Vector2i &tile) {
        std::mt19937 rng{std::random_device{}()};
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, img.width);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, img.height);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Vector3 color = {0, 0, 0};
                for (int i = 0; i < scene.options.spp; i++){
                    Ray r = {cam.lookfrom,
                            normalize(
                            u * ((x + random_double(rng)) / img.width - Real(0.5)) * viewport_width +
                            v * ((y + random_double(rng)) / img.height - Real(0.5)) * viewport_height -
                            w),
                            c_EPSILON,
                            infinity<Real>()};
                    color += trace_ray_MIS(scene, r, rng);
                }
                img(x, img.height - y - 1) = color / Real(scene.options.spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    std::cout << std::endl << "Finish building rendering. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}

Image3 hw_4_4(const std::vector<std::string> &params) {
    if (params.size() < 1) {
        return Image3(0, 0);
    }

    int max_depth = 50;
    std::string filename;
    for (int i = 0; i < (int)params.size(); i++) {
        if (params[i] == "-max_depth") {
            max_depth = std::stoi(params[++i]);
        } else if (filename.empty()) {
            filename = params[i];
        }
    }

    Timer timer;
    std::cout << "Parsing and constructing scene " << params[0] << "." << std::endl;
    tick(timer);
    ParsedScene pscene = parse_scene(params[0]);
    std::cout << "Scene parsing done. Took " << tick(timer) << " seconds." << std::endl;
    UNUSED(pscene);

    Scene scene(pscene);
    scene.options.max_depth = max_depth;

    Image3 img(scene.width, scene.height);

    Camera &cam = scene.camera;

    Real theta = cam.vfov / 180 * c_PI;
    Real h = tan(theta/2);
    Real viewport_height = 2.0 * h;
    Real viewport_width = viewport_height / img.height * img.width;

    Vector3 w = normalize(cam.lookfrom - cam.lookat);
    Vector3 u = normalize(cross(cam.up, w));
    Vector3 v = cross(w, u);

    // Build BVH
    std::cout << "Building BVH..." << std::endl;
    tick(timer);
    build_bvh(scene);
    std::cout << "Finish building BVH. Took " << tick(timer) << " seconds." << std::endl;

    constexpr int tile_size = 16;
    int num_tiles_x = (img.width + tile_size - 1) / tile_size;
    int num_tiles_y = (img.height + tile_size - 1) / tile_size;
    ProgressReporter reporter(num_tiles_x * num_tiles_y);
    
    std::cout << "Rendering..." << std::endl;
    tick(timer);
    parallel_for([&](const Vector2i &tile) {
        std::mt19937 rng{std::random_device{}()};
        int x0 = tile[0] * tile_size;
        int x1 = min(x0 + tile_size, img.width);
        int y0 = tile[1] * tile_size;
        int y1 = min(y0 + tile_size, img.height);
        for (int y = y0; y < y1; y++) {
            for (int x = x0; x < x1; x++) {
                Vector3 color = {0, 0, 0};
                for (int i = 0; i < scene.options.spp; i++){
                    Ray r = {cam.lookfrom,
                            normalize(
                            u * ((x + random_double(rng)) / img.width - Real(0.5)) * viewport_width +
                            v * ((y + random_double(rng)) / img.height - Real(0.5)) * viewport_height -
                            w),
                            c_EPSILON,
                            infinity<Real>()};
                    color += trace_ray_MIS_power(scene, r, rng);
                }
                img(x, img.height - y - 1) = color / Real(scene.options.spp);
            }
        }
        reporter.update(1);
    }, Vector2i(num_tiles_x, num_tiles_y));
    std::cout << std::endl << "Finish building rendering. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}
