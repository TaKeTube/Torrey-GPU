#include "material.h"

struct sample_bsdf_op{
    std::optional<SampleRecord> operator()(const Diffuse &m) const;
    std::optional<SampleRecord> operator()(const Mirror &m) const;
    std::optional<SampleRecord> operator()(const Plastic &m) const;
    std::optional<SampleRecord> operator()(const Phong &m) const;
    std::optional<SampleRecord> operator()(const BlinnPhong &m) const;
    std::optional<SampleRecord> operator()(const BlinnPhongMicrofacet &m) const;

    const Vector3 &dir_in;
    const Intersection &v;
    const TexturePool &texture_pool;
    std::mt19937 &rng;
};

struct sample_bsdf_pdf_op{
    Real operator()(const Diffuse &m) const;
    Real operator()(const Mirror &m) const;
    Real operator()(const Plastic &m) const;
    Real operator()(const Phong &m) const;
    Real operator()(const BlinnPhong &m) const;
    Real operator()(const BlinnPhongMicrofacet &m) const;

    const Vector3 &dir_in;
    const Vector3 &dir_out;
    const Intersection &v;
    const TexturePool &pool;
};

struct eval_material_op{
    Vector3 operator()(const Diffuse &m) const;
    Vector3 operator()(const Mirror &m) const;
    Vector3 operator()(const Plastic &m) const;
    Vector3 operator()(const Phong &m) const;
    Vector3 operator()(const BlinnPhong &m) const;
    Vector3 operator()(const BlinnPhongMicrofacet &m) const;

    const Vector3 &dir_in;
    const SampleRecord &record;
    const Intersection &v;
    const TexturePool &texture_pool;
};

#include "materials/diffuse.inl"
#include "materials/mirror.inl"
#include "materials/plastic.inl"
#include "materials/phong.inl"
#include "materials/blinn_phong.inl"
#include "materials/blinn_phong_microfacet.inl"

std::optional<SampleRecord> sample_bsdf(const Material &material,
                                        const Vector3 &dir_in,
                                        const Intersection &v,
                                        const TexturePool &pool,
                                        std::mt19937 &rng){
    return std::visit(sample_bsdf_op{dir_in, v, pool, rng}, material);
}

Real get_bsdf_pdf(const Material &material,
                  const Vector3 &dir_in,
                  const Vector3 &dir_out,
                  const Intersection &v,
                  const TexturePool &pool){
    return std::visit(sample_bsdf_pdf_op{dir_in, dir_out, v, pool}, material);
}

Vector3 eval(const Material &material,
             const Vector3 &dir_in,
             const SampleRecord &record,
             const Intersection &v,
             const TexturePool &pool){
    return std::visit(eval_material_op{dir_in, record, v, pool}, material);
}