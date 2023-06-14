#include "material.h"

__device__ std::optional<SampleRecord> sample_bsdf_Diffuse(const Diffuse &m, const Vector3 &dir_in, const Intersection &v, const TexturePool &texture_pool, std::mt19937 &rng);
__device__ std::optional<SampleRecord> sample_bsdf_Mirror(const Mirror &m, const Vector3 &dir_in, const Intersection &v, const TexturePool &texture_pool, std::mt19937 &rng);
__device__ std::optional<SampleRecord> sample_bsdf_Plastic(const Plastic &m, const Vector3 &dir_in, const Intersection &v, const TexturePool &texture_pool, std::mt19937 &rng);
__device__ std::optional<SampleRecord> sample_bsdf_Phong(const BlinnPhong &m, const Vector3 &dir_in, const Intersection &v, const TexturePool &texture_pool, std::mt19937 &rng);
__device__ std::optional<SampleRecord> sample_bsdf_BlinnPhong(const BlinnPhong &m, const Vector3 &dir_in, const Intersection &v, const TexturePool &texture_pool, std::mt19937 &rng);
__device__ std::optional<SampleRecord> sample_bsdf_BlinnPhongMicrofacet(const BlinnPhongMicrofacet &m, const Vector3 &dir_in, const Intersection &v, const TexturePool &texture_pool, std::mt19937 &rng);

__device__ Real sample_bsdf_pdf_Diffuse(const Diffuse &m, const Vector3 &dir_in, const Vector3 &dir_out, const Intersection &v, const TexturePool &texture_pool);
__device__ Real sample_bsdf_pdf_Mirror(const Mirror &m, const Vector3 &dir_in, const Vector3 &dir_out, const Intersection &v, const TexturePool &texture_pool);
__device__ Real sample_bsdf_pdf_Plastic(const Plastic &m, const Vector3 &dir_in, const Vector3 &dir_out, const Intersection &v, const TexturePool &texture_pool);
__device__ Real sample_bsdf_pdf_Phong(const Phong &m, const Vector3 &dir_in, const Vector3 &dir_out, const Intersection &v, const TexturePool &texture_pool);
__device__ Real sample_bsdf_pdf_BlinnPhong(const BlinnPhong &m, const Vector3 &dir_in, const Vector3 &dir_out, const Intersection &v, const TexturePool &texture_pool);
__device__ Real sample_bsdf_pdf_BlinnPhongMicrofacet(const BlinnPhongMicrofacet &m, const Vector3 &dir_in, const Vector3 &dir_out, const Intersection &v, const TexturePool &texture_pool);


__device__ Vector3 eval_material_Diffuse(const Diffuse &m, const Vector3 &dir_in, const SampleRecord &record, const Intersection &v, const TexturePool &texture_pool);
__device__ Vector3 eval_material_Mirror(const Mirror &m, const Vector3 &dir_in, const SampleRecord &record, const Intersection &v, const TexturePool &texture_pool);
__device__ Vector3 eval_material_Plastic(const Plastic &m, const Vector3 &dir_in, const SampleRecord &record, const Intersection &v, const TexturePool &texture_pool);
__device__ Vector3 eval_material_Phong(const Phong &m, const Vector3 &dir_in, const SampleRecord &record, const Intersection &v, const TexturePool &texture_pool);
__device__ Vector3 eval_material_BlinnPhong(const BlinnPhong &m, const Vector3 &dir_in, const SampleRecord &record, const Intersection &v, const TexturePool &texture_pool);
__device__ Vector3 eval_material_BlinnPhongMicrofacet(const BlinnPhongMicrofacet &m, const Vector3 &dir_in, const SampleRecord &record, const Intersection &v, const TexturePool &texture_pool);

#include "materials/diffuse.inl"
#include "materials/mirror.inl"
#include "materials/plastic.inl"
#include "materials/phong.inl"
#include "materials/blinn_phong.inl"
#include "materials/blinn_phong_microfacet.inl"

__device__ std::optional<SampleRecord> sample_bsdf(const Material &material,
                                                   const Vector3 &dir_in,
                                                   const Intersection &v,
                                                   const TexturePool &texture_pool,
                                                   std::mt19937 &rng) {
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

__device__ Real get_bsdf_pdf(const Material &material,
                             const Vector3 &dir_in,
                             const Vector3 &dir_out,
                             const Intersection &v,
                             const TexturePool &texture_pool) {
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

__device__ Vector3 eval(const Material &material,
                        const Vector3 &dir_in,
                        const SampleRecord &record,
                        const Intersection &v,
                        const TexturePool &texture_pool) {
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