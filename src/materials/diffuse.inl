std::optional<SampleRecord> sample_bsdf_op::operator()(const Diffuse &m) const {
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

Real sample_bsdf_pdf_op::operator()(const Diffuse &m) const {
    if (dot(v.geo_normal, dir_out) < 0) 
        return Real(0);
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
    return fmax(dot(n, dir_out), Real(0)) / c_PI;
}

Vector3 eval_material_op::operator()(const Diffuse &m) const {
    if (dot(v.geo_normal, dir_in) < 0 || dot(v.geo_normal, record.dir_out) < 0)
        return {Real(0), Real(0), Real(0)};
    Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
    const Vector3& Kd = eval(m.reflectance, v.uv, texture_pool);
    return Kd * fmax(dot(n, record.dir_out), Real(0)) / c_PI;
}