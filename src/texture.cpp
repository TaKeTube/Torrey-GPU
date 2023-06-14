#include "texture.h"

__device__ Vector3 eval_texture_Constant(const ConstTexture &t){
    return t.value;
}

__device__ Vector3 eval_texture_Image(const ImageTexture &t, const Vector2 &uv, const TexturePool &pool){
    const Image3& img = pool.image3s.at(t.texture_id);
    Real x = img.width * modulo(t.uscale * uv.x + t.uoffset, Real(1));
    Real y = img.height * modulo(t.vscale * uv.y + t.voffset, Real(1));
    // Bilinear Interpolation
    int x1 = floor(x);
    int x2 = (x1 + 1) == img.width ? 0 : (x1 + 1);
    int y1 = floor(y);
    int y2 = (y1 + 1) == img.height ? 0 : (y1 + 1);
    Vector3 q11 = img(x1, y1);
    Vector3 q12 = img(x1, y2);
    Vector3 q21 = img(x2, y1);
    Vector3 q22 = img(x2, y2);
    if(x1 == x2)
        x2 += 1;
    if(y1 == y2)
        y2 += 1;
    return (q11*(x2-x)*(y2-y)+q21*(x-x1)*(y2-y)+q12*(x2-x)*(y-y1)+q22*(x-x1)*(y-y1))/Real((x2-x1)*(y2-y1));
    // return img(floor(x), floor(y));
}