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

struct eval_texture_op {
    Vector3 operator()(const ConstTexture &t) const;
    Vector3 operator()(const ImageTexture &t) const;

    const Vector2 &uv;
    const TexturePool &pool;
};

inline Vector3 eval(const Texture &texture, const Vector2 &uv, const TexturePool &pool) {
    return std::visit(eval_texture_op{uv, pool}, texture);
}