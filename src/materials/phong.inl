__device__ inline std::optional<SampleRecord> sample_bsdf_Phong(const Phong &m,
                                                         const Vector3 &dir_in,
                                                         const Intersection &v,
                                                         const DeviceTexturePool &texture_pool,
                                                         RNGf &rng) {
    if (dot(v.geo_normal, dir_in) < 0) {
        return {};
    }
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;

    Real u1 = random_double(rng);
    Real u2 = random_double(rng);

    Real reciprocal_alpha_1 = 1 / (m.exponent + 1);
    Real phi = c_TWOPI * u2;
    Real sqrt_u1 = sqrt(std::clamp(1 - pow(u1, 2 * reciprocal_alpha_1), Real(0), Real(1)));
    Vector3 local_out = normalize(Vector3{
        cos(phi) * sqrt_u1, 
        sin(phi) * sqrt_u1,
        std::clamp(pow(u1, reciprocal_alpha_1), Real(0), Real(1))
    });

    Vector3 reflect_dir = normalize(-dir_in + 2*dot(dir_in, n) * n);
    SampleRecord record;
    record.dir_out = normalize(to_world(reflect_dir, local_out));
    if (dot(v.geo_normal, record.dir_out) < 0) 
        record.pdf = Real(0);
    else
        record.pdf = fmax(Real(0), (m.exponent + 1) / c_TWOPI * pow(dot(reflect_dir, record.dir_out), m.exponent));
        // record.pdf = Real(1);
    return record;
}

__device__ inline Real sample_bsdf_pdf_Phong(const Phong &m,
                                      const Vector3 &dir_in,
                                      const Vector3 &dir_out,
                                      const Intersection &v,
                                      const DeviceTexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_out) < 0) 
        return Real(0);
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;

    Vector3 reflect_dir = normalize(-dir_in + 2*dot(dir_in, n) * n);
    if (dot(v.geo_normal, dir_out) < 0) 
        return Real(0);
    else
        return fmax(Real(0), (m.exponent + 1) / c_TWOPI * pow(dot(reflect_dir, dir_out), m.exponent));
}

__device__ inline Vector3 eval_material_Phong(const Phong &m,
                                       const Vector3 &dir_in,
                                       const SampleRecord &record,
                                       const Intersection &v,
                                       const DeviceTexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_in) < 0 || dot(v.geo_normal, record.dir_out) < 0)
        return {Real(0), Real(0), Real(0)};
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;

    Vector3 reflect_dir = normalize(-dir_in + 2*dot(dir_in, n) * n);
    const Vector3& Ks = eval(m.reflectance, v.uv, texture_pool);
    if (dot(n, record.dir_out) <= 0)
        return {Real(0), Real(0), Real(0)};
    else
        return Ks*(m.exponent+1)/c_TWOPI*pow(fmax(dot(record.dir_out, reflect_dir), Real(0)), m.exponent);
        // return Ks;
}