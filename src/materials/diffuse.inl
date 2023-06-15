__device__ inline std::optional<SampleRecord> sample_bsdf_Diffuse(const Diffuse &m, 
                                                           const Vector3 &dir_in,
                                                           const Intersection &v,
                                                           const DeviceTexturePool &texture_pool,
                                                           RNGf &rng) {
    if (dot(v.geo_normal, dir_in) < 0) {
        return {};
    }
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
    SampleRecord record;
    record.dir_out = to_world(n, sample_hemisphere_cos(rng));
    if (dot(v.geo_normal, record.dir_out) < 0) 
        record.pdf = Real(0);
    else
        record.pdf = fmax(dot(n, record.dir_out), Real(0)) / c_PI;
    return record;
}

__device__ inline Real sample_bsdf_pdf_Diffuse(const Diffuse &m, 
                                        const Vector3 &dir_in,
                                        const Vector3 &dir_out,
                                        const Intersection &v,
                                        const DeviceTexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_out) < 0) 
        return Real(0);
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
    return fmax(dot(n, dir_out), Real(0)) / c_PI;
}

__device__ inline Vector3 eval_material_Diffuse(const Diffuse &m,
                                         const Vector3 &dir_in,
                                         const SampleRecord &record,
                                         const Intersection &v,
                                         const DeviceTexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_in) < 0 || dot(v.geo_normal, record.dir_out) < 0)
        return {Real(0), Real(0), Real(0)};
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
    const Vector3& Kd = eval(m.reflectance, v.uv, texture_pool);
    return Kd * fmax(dot(n, record.dir_out), Real(0)) / c_PI;
}