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

__device__ inline Vector3 eval_texture_Constant(const ConstTexture &t){
    return t.value;
}

__device__ inline Vector3 eval_texture_Image(const ImageTexture &t, const Vector2 &uv, const DeviceTexturePool &pool){
    const DeviceImage3& img = pool.image3s[t.texture_id];
    Real x = img.width * modulo(t.uscale * uv.x + t.uoffset, Real(1));
    Real y = img.height * modulo(t.vscale * uv.y + t.voffset, Real(1));
    // Bilinear Interpolation
    int x1 = floor(x);
    int x2 = (x1 + 1) == img.width ? 0 : (x1 + 1);
    int y1 = floor(y);
    int y2 = (y1 + 1) == img.height ? 0 : (y1 + 1);
    Vector3 q11 = img.data[y1 * img.width + x1]; 
    Vector3 q12 = img.data[y2 * img.width + x1]; 
    Vector3 q21 = img.data[y1 * img.width + x2];
    Vector3 q22 = img.data[y2 * img.width + x2];
    if(x1 == x2)
        x2 += 1;
    if(y1 == y2)
        y2 += 1;
    return (q11*(x2-x)*(y2-y)+q21*(x-x1)*(y2-y)+q12*(x2-x)*(y-y1)+q22*(x-x1)*(y-y1))/Real((x2-x1)*(y2-y1));
    // return img(floor(x), floor(y));
}

__device__ inline Vector3 eval(const Texture &texture, const Vector2 &uv, const DeviceTexturePool &pool) {
    if(auto* t = std::get_if<ConstTexture>(&texture))
        return eval_texture_Constant(*t);
    else if(auto *t = std::get_if<ImageTexture>(&texture))
        return eval_texture_Image(*t, uv, pool);
    else
        return Vector3{0.0, 0.0, 0.0};
}
