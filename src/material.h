#pragma once
#include <optional>
#include "vector.h"
#include "intersection.h"
#include "texture.h"

struct Diffuse {
    Texture reflectance;
};

struct Mirror {
    Texture reflectance;
    Real eta = 1;
};

struct Plastic {
    Texture reflectance;
    Real eta = 1;
};

struct Phong {
    Texture reflectance; // Ks
    Real exponent; // alpha
};

struct BlinnPhong {
    Texture reflectance; // Ks
    Real exponent; // alpha
};

struct BlinnPhongMicrofacet {
    Texture reflectance; // Ks
    Real exponent; // alpha
};

using Material = std::variant<Diffuse,
                              Mirror,
                              Plastic,
                              Phong,
                              BlinnPhong,
                              BlinnPhongMicrofacet>;

struct SampleRecord {
    Vector3 dir_out;
    Real pdf;
};

__device__ Vector3 eval(
    const Material &material,
    const Vector3 &dir_in,
    const SampleRecord &record,
    const Intersection &v,
    const DeviceTexturePool &pool);

__device__ std::optional<SampleRecord> sample_bsdf(
    const Material &material,
    const Vector3 &dir_in,
    const Intersection &v,
    const DeviceTexturePool &pool,
    RNGf &rng);

__device__ Real get_bsdf_pdf(
    const Material &material,
    const Vector3 &dir_in,
    const Vector3 &dir_out,
    const Intersection &v,
    const DeviceTexturePool &pool);

__device__ inline Vector3 sample_hemisphere_cos(RNGf& rng) {
    Real u1 = random_double(rng);
    Real u2 = random_double(rng);
    
    Real phi = c_TWOPI * u2;
    Real sqrt_u1 = sqrt(std::clamp(u1, Real(0), Real(1)));
    return Vector3{
        cos(phi) * sqrt_u1, 
        sin(phi) * sqrt_u1,
        sqrt(std::clamp(1 - u1, Real(0), Real(1)))
    };
}

__device__ inline Real compute_blinn_phong_G_hat(Vector3 omega, Vector3 n, Real alpha) {
    Real odn = dot(omega, n);
    Real a = sqrt(0.5 * alpha + 1)/sqrt(1/(odn * odn) - 1);
    Real a2 = a*a;
    Real G_hat = a < 1.6 ? (3.535*a+2.181*a2)/(1+2.276*a+2.577*a2) : 1;
    return G_hat;
}

#include "materials/diffuse.inl"
#include "materials/mirror.inl"
#include "materials/plastic.inl"
#include "materials/phong.inl"
#include "materials/blinn_phong.inl"
#include "materials/blinn_phong_microfacet.inl"

__device__ inline std::optional<SampleRecord> sample_bsdf(const Material &material,
                                                   const Vector3 &dir_in,
                                                   const Intersection &v,
                                                   const DeviceTexturePool &texture_pool,
                                                   RNGf &rng) {
    if(auto *m = std::get_if<Diffuse>(&material))
        return sample_bsdf_Diffuse(*m, dir_in, v, texture_pool, rng);
    else if(auto *m = std::get_if<Mirror>(&material))
        return sample_bsdf_Mirror(*m, dir_in, v, texture_pool, rng);
    else if(auto *m = std::get_if<Plastic>(&material))
        return sample_bsdf_Plastic(*m, dir_in, v, texture_pool, rng);
    else if(auto *m = std::get_if<Phong>(&material))
        return sample_bsdf_Phong(*m, dir_in, v, texture_pool, rng);
    else if(auto *m = std::get_if<BlinnPhong>(&material))
        return sample_bsdf_BlinnPhong(*m, dir_in, v, texture_pool, rng);
    else if(auto *m = std::get_if<BlinnPhongMicrofacet>(&material))
        return sample_bsdf_BlinnPhongMicrofacet(*m, dir_in, v, texture_pool, rng);
    else
        return {};
}

__device__ inline Real get_bsdf_pdf(const Material &material,
                             const Vector3 &dir_in,
                             const Vector3 &dir_out,
                             const Intersection &v,
                             const DeviceTexturePool &texture_pool) {
    if(auto *m = std::get_if<Diffuse>(&material))
        return sample_bsdf_pdf_Diffuse(*m, dir_in, dir_out, v, texture_pool);
    else if(auto *m = std::get_if<Mirror>(&material))
        return sample_bsdf_pdf_Mirror(*m, dir_in, dir_out, v, texture_pool);
    else if(auto *m = std::get_if<Plastic>(&material))
        return sample_bsdf_pdf_Plastic(*m, dir_in, dir_out, v, texture_pool);
    else if(auto *m = std::get_if<Phong>(&material))
        return sample_bsdf_pdf_Phong(*m, dir_in, dir_out, v, texture_pool);
    else if(auto *m = std::get_if<BlinnPhong>(&material))
        return sample_bsdf_pdf_BlinnPhong(*m, dir_in, dir_out, v, texture_pool);
    else if(auto *m = std::get_if<BlinnPhongMicrofacet>(&material))
        return sample_bsdf_pdf_BlinnPhongMicrofacet(*m, dir_in, dir_out, v, texture_pool);
    else
        return {};
}

__device__ inline Vector3 eval(const Material &material,
                        const Vector3 &dir_in,
                        const SampleRecord &record,
                        const Intersection &v,
                        const DeviceTexturePool &texture_pool) {
    if(auto *m = std::get_if<Diffuse>(&material))
        return eval_material_Diffuse(*m, dir_in, record, v, texture_pool);
    else if(auto *m = std::get_if<Mirror>(&material))
        return eval_material_Mirror(*m, dir_in, record, v, texture_pool);
    else if(auto *m = std::get_if<Plastic>(&material))
        return eval_material_Plastic(*m, dir_in, record, v, texture_pool);
    else if(auto *m = std::get_if<Phong>(&material))
        return eval_material_Phong(*m, dir_in, record, v, texture_pool);
    else if(auto *m = std::get_if<BlinnPhong>(&material))
        return eval_material_BlinnPhong(*m, dir_in, record, v, texture_pool);
    else if(auto *m = std::get_if<BlinnPhongMicrofacet>(&material))
        return eval_material_BlinnPhongMicrofacet(*m, dir_in, record, v, texture_pool);
    else
        return {};
}