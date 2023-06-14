__device__ std::optional<SampleRecord> sample_bsdf_BlinnPhongMicrofacet(const BlinnPhongMicrofacet &m,
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
    Vector3 local_h = normalize(Vector3{
        cos(phi) * sqrt_u1, 
        sin(phi) * sqrt_u1,
        std::clamp(pow(u1, reciprocal_alpha_1), Real(0), Real(1))
    });

    Vector3 h = normalize(to_world(n, local_h));

    SampleRecord record;
    record.dir_out = normalize(-dir_in + 2*dot(dir_in, h) * h);
    if (dot(v.geo_normal, record.dir_out) <= 0 || dot(h, n) <= 0 || dot(record.dir_out, h) <= 0) 
        record.pdf = Real(0);
    else
        record.pdf = (m.exponent + 1) * 0.25 * c_INVTWOPI * pow(std::clamp(dot(n, h), Real(0), Real(1)), m.exponent) / dot(record.dir_out, h);
    return record;
}

__device__ Real sample_bsdf_pdf_BlinnPhongMicrofacet(const BlinnPhongMicrofacet &m,
                                                     const Vector3 &dir_in,
                                                     const Vector3 &dir_out,
                                                     const Intersection &v,
                                                     const DeviceTexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_out) < 0) 
        return Real(0);
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;

    Vector3 h = normalize(dir_out + dir_in);
    if (dot(v.geo_normal, dir_out) <= 0 || dot(h, n) <= 0 || dot(dir_out, h) <= 0) 
        return Real(0);
    else
        return (m.exponent + 1) * 0.25 * c_INVTWOPI * pow(std::clamp(dot(n, h), Real(0), Real(1)), m.exponent) / dot(dir_out, h);
}

__device__ Vector3 eval_material_BlinnPhongMicrofacet(const BlinnPhongMicrofacet &m,
                                                      const Vector3 &dir_in,
                                                      const SampleRecord &record,
                                                      const Intersection &v,
                                                      const DeviceTexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_in) < 0 || dot(v.geo_normal, record.dir_out) < 0)
        return {Real(0), Real(0), Real(0)};
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;

    Vector3 h = normalize(record.dir_out + dir_in);

    if (dot(n, record.dir_out) <= 0 || dot(record.dir_out, h) <= 0 || dot(dir_in, h) <= 0)
        return {Real(0), Real(0), Real(0)};
    else{
        const Vector3& Ks = eval(m.reflectance, v.uv, texture_pool);
        
        Vector3 Fh = Ks + (1 - Ks) * pow(1 - dot(h, record.dir_out), 5);
        Real Dh = (m.exponent + 2) * c_INVTWOPI * pow(std::clamp(dot(n, h), Real(0), Real(1)), m.exponent);
        Real G = compute_blinn_phong_G_hat(record.dir_out, n, m.exponent) * compute_blinn_phong_G_hat(dir_in, n, m.exponent);

        return Fh * Dh * G * 0.25 / dot(n, dir_in);
    }
}