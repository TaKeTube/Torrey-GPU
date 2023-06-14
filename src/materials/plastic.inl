__device__ std::optional<SampleRecord> sample_bsdf_Plastic(const Plastic &m,
                                                           const Vector3 &dir_in,
                                                           const Intersection &v,
                                                           const TexturePool &texture_pool,
                                                           RNGf &rng) {
    if (dot(v.geo_normal, dir_in) < 0) {
        return {};
    }
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
    SampleRecord record;

    Vector3 reflect_dir = -dir_in + 2*dot(dir_in, n) * n;
    Real eta = m.eta;
    Real F0 = pow((eta - 1)/(eta + 1), 2);
    Real F = F0 + (1 - F0) * pow(1 - dot(n, reflect_dir), 5);
    
    Real u = random_double(rng);
    if(u <= F){
        record.dir_out = reflect_dir;
        record.pdf = Real(1);
    }else{
        record.dir_out = to_world(n, sample_hemisphere_cos(rng));
        if (dot(v.geo_normal, record.dir_out) < 0) 
            record.pdf = Real(0);
        else
            record.pdf = fmax(dot(n, record.dir_out), Real(0)) / c_PI;
    }
    return record;
}

__device__ Real sample_bsdf_pdf_Plastic(const Plastic &m,
                                        const Vector3 &dir_in,
                                        const Vector3 &dir_out,
                                        const Intersection &v,
                                        const TexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_out) < 0) 
        return Real(0);
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;

    Real eta = m.eta;
    Real F0 = pow((eta - 1)/(eta + 1), 2);
    Real F = F0 + (1 - F0) * pow(1 - dot(n, dir_out), 5);
    return length(normalize(dir_in + dir_out) - n) < c_EPSILON ? Real(F) : (1 - F) * fmax(dot(n, dir_out), Real(0)) / c_PI;
}

__device__ Vector3 eval_material_Plastic(const Plastic &m,
                                         const Vector3 &dir_in,
                                         const SampleRecord &record,
                                         const Intersection &v,
                                         const TexturePool &texture_pool) {
    if (dot(v.geo_normal, dir_in) < 0 || dot(v.geo_normal, record.dir_out) < 0)
        return {Real(0), Real(0), Real(0)};
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;

    if(record.pdf == Real(1)){
        return Vector3{Real(1), Real(1), Real(1)};
    }else{
        const Vector3& Kd = eval(m.reflectance, v.uv, texture_pool);
        return Kd * fmax(dot(n, record.dir_out), Real(0)) / c_PI;
    }
}