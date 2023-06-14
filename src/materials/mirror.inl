__device__ std::optional<SampleRecord> sample_bsdf_Mirror(const Mirror &m,
                                                          const Vector3 &dir_in,
                                                          const Intersection &v,
                                                          const TexturePool &texture_pool,
                                                          RNGf &rng) {
    if (dot(v.geo_normal, dir_in) < 0) {
        return {};
    }
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
    SampleRecord record;
    record.dir_out = -dir_in + 2*dot(dir_in, n) * n;
    record.pdf = 1;
    return record;
}

__device__ Real sample_bsdf_pdf_Mirror(const Mirror &m,
                                       const Vector3 &dir_in,
                                       const Vector3 &dir_out,
                                       const Intersection &v,
                                       const TexturePool &texture_pool) {
    if(length(normalize(dir_in + dir_out) - v.shading_normal) < c_EPSILON)
        return Real(1);
    return Real(0);
}

__device__ Vector3 eval_material_Mirror(const Mirror &m,
                                        const Vector3 &dir_in,
                                        const SampleRecord &record,
                                        const Intersection &v,
                                        const TexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_in) < 0 || dot(v.geo_normal, record.dir_out) < 0)
        return {Real(0), Real(0), Real(0)};
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
    const Vector3& F0 = eval(m.reflectance, v.uv, texture_pool);
    Vector3 F = F0 + (1 - F0) * pow(1 - dot(n, record.dir_out), 5);
    return F;
}