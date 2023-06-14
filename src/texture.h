#pragma once
#include "image.h"
#include "vector.h"
#include <variant>
#include <map>
#include <string>

struct TexturePool {
    std::map<std::string, int> image1s_map;
    std::map<std::string, int> image3s_map;

    std::vector<Image1> image1s;
    std::vector<Image3> image3s;
};

struct DeviceImage1 {
    int width;
    int height;
    Real* data;
};

struct DeviceImage3 {
    int width;
    int height;
    Vector3* data;
};

struct DeviceTexturePool {
    DeviceImage1* image1s;
    DeviceImage3* image3s;
};

struct ImageTexture {
    int texture_id;
    Real uscale, vscale;
    Real uoffset, voffset;
};

struct ConstTexture
{
    Vector3 value;
};

using Texture = std::variant<ConstTexture, ImageTexture>;

__device__ Vector3 eval_texture_Constant(const ConstTexture &t);
__device__ Vector3 eval_texture_Image(const ImageTexture &t, const Vector2 &uv, const DeviceTexturePool &pool);

__device__ inline Vector3 eval(const Texture &texture, const Vector2 &uv, const DeviceTexturePool &pool) {
    if(auto* t = std::get_if<ConstTexture>(&texture))
        return eval_texture_Constant(*t);
    else if(auto *t = std::get_if<ImageTexture>(&texture))
        return eval_texture_Image(*t, uv, pool);
}
