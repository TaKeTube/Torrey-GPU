#include "render.h"
#include "parse_scene.h"
#include "scene.h"
#include "timer.h"
#include "parallel.h"
#include "progressreporter.h"

__global__ void render_kernel(deviceScene scene, sceneInfo scene_info, Vector3 *img)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = bx * TILE_WIDTH + tx;
    int y = by * TILE_WIDTH + ty;

    if (x >= scene_info.width || y >= scene_info.height)
        return;

    // random number initialization
    RNG<float> rng(y * scene_info.width + x);

    // preparation for ray generation
    Camera &cam = scene_info.camera;
    RenderOptions &options = scene_info.options;
    Real theta = cam.vfov / 180 * c_PI;
    Real h = tan(theta / 2);
    Real viewport_height = 2.0 * h;
    Real viewport_width = viewport_height / scene_info.height * scene_info.width;

    Vector3 w = normalize(cam.lookfrom - cam.lookat);
    Vector3 u = normalize(cross(cam.up, w));
    Vector3 v = cross(w, u);

    // Trace ray
    Vector3 color = {0, 0, 0};
    for (int i = 0; i < options.spp; i++)
    {
        Ray r = {cam.lookfrom,
                 normalize(
                     u * ((x + random_double(rng)) / scene_info.width - Real(0.5)) * viewport_width +
                     v * ((y + random_double(rng)) / scene_info.height - Real(0.5)) * viewport_height -
                     w),
                 c_EPSILON,
                 infinity<Real>()};
        color += trace_ray(scene, scene_info, r, rng);
    }
    color /= Real(options.spp);

    // Set output pixel
    int img_pos = (scene_info.height - y - 1) * scene_info.width + x;
    img[img_pos] = color;
}

Image3 render(const std::vector<std::string> &params)
{
    if (params.size() < 1)
    {
        return Image3(0, 0);
    }

    int max_depth = 50;
    std::string filename;
    for (int i = 0; i < (int)params.size(); i++)
    {
        if (params[i] == "-max_depth")
        {
            max_depth = std::stoi(params[++i]);
        }
        else if (filename.empty())
        {
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
    Real h = tan(theta / 2);
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

    std::cout << "Rendering..." << std::endl;
    tick(timer);

    // Device Memory Init
    auto [device_scene, free_info] = device_scene_init(scene);
    Vector3* deviceImg;
    cudaMalloc((void **)&deviceImg, img.height * img.width * sizeof(Vector3));

    // Kernel Init
    dim3 DimGrid(ceil(((float)img.width) / TILE_WIDTH), ceil(((float)img.height) / TILE_WIDTH), 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    // Run kernel
    render_kernel<<<DimGrid, DimBlock>>>(scene, sceneInfo, deviceImg);

    // Checking Errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();

    // Copy the device image back to the host
    cudaMemcpy(img.data.data(), deviceImg, img.height * img.width * sizeof(Vector3), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceImg);
    device_scene_destruct(device_scene, free_info);

    std::cout << std::endl
              << "Finish building rendering. Took " << tick(timer) << " seconds." << std::endl;

    return img;
}