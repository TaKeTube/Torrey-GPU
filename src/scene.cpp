#include "scene.h"

Scene::Scene(){}

Scene::Scene(const ParsedScene &scene) : camera(from_parsed_camera(scene.camera)),
                                            width(scene.camera.width),
                                            height(scene.camera.height),
                                            background_color(scene.background_color)
{
    options.spp = scene.samples_per_pixel;

    // Extract triangle meshes from the parsed scene.
    int tri_mesh_count = 0;
    for (const ParsedShape &parsed_shape : scene.shapes)
    {
        if (std::get_if<ParsedTriangleMesh>(&parsed_shape))
        {
            tri_mesh_count++;
        }
    }
    meshes.resize(tri_mesh_count);
    // Extract the shapes
    tri_mesh_count = 0;
    for (int i = 0; i < (int)scene.shapes.size(); i++)
    {
        const ParsedShape &parsed_shape = scene.shapes[i];
        if (auto *sph = std::get_if<ParsedSphere>(&parsed_shape))
        {
            shapes.push_back(
                Sphere{{sph->material_id, sph->area_light_id},
                        sph->position,
                        sph->radius});
        }
        else if (auto *parsed_mesh = std::get_if<ParsedTriangleMesh>(&parsed_shape))
        {
            meshes[tri_mesh_count] = TriangleMesh{
                {parsed_mesh->material_id, parsed_mesh->area_light_id},
                parsed_mesh->positions,
                parsed_mesh->indices,
                parsed_mesh->normals,
                parsed_mesh->uvs};
            // Extract all the individual triangles
            for (int face_index = 0; face_index < (int)parsed_mesh->indices.size(); face_index++)
            {
                shapes.push_back(Triangle{face_index, tri_mesh_count, parsed_mesh->area_light_id});
            }
            tri_mesh_count++;
        }
        else
        {
            assert(false);
        }
    }
    // Copy the materials
    for (const ParsedMaterial &parsed_mat : scene.materials)
    {
        if (auto *diffuse = std::get_if<ParsedDiffuse>(&parsed_mat))
        {
            if (auto *color = std::get_if<Vector3>(&diffuse->reflectance)) {
                ConstTexture texture = {*color};
                materials.push_back(Diffuse{texture});
            } else if (auto *color = std::get_if<ParsedImageTexture>(&diffuse->reflectance)) {
                std::string texture_name = color->filename.string();
                int texture_id;
                auto it = textures.image3s_map.find(texture_name);
                if (it != textures.image3s_map.end()) {
                    texture_id = textures.image3s_map.at(texture_name);
                } else {
                    texture_id = textures.image3s.size();
                    textures.image3s_map.emplace(texture_name, texture_id);
                    textures.image3s.push_back(imread3(color->filename));
                }
                ImageTexture texture = {
                    texture_id,
                    color->uscale, color->vscale,
                    color->uoffset, color->voffset,
                };
                materials.push_back(Diffuse{texture});
            }
        }
        else if (auto *mirror = std::get_if<ParsedMirror>(&parsed_mat))
        {
            if (auto *color = std::get_if<Vector3>(&mirror->reflectance)) {
                ConstTexture texture = {*color};
                materials.push_back(Mirror{texture});
            } else if (auto *color = std::get_if<ParsedImageTexture>(&mirror->reflectance)) {
                std::string texture_name = color->filename.string();
                int texture_id;
                auto it = textures.image3s_map.find(texture_name);
                if (it != textures.image3s_map.end()) {
                    texture_id = textures.image3s_map.at(texture_name);
                } else {
                    texture_id = textures.image3s.size();
                    textures.image3s_map.emplace(texture_name, texture_id);
                    textures.image3s.push_back(imread3(color->filename));
                }
                ImageTexture texture = {
                    texture_id,
                    color->uscale, color->vscale,
                    color->uoffset, color->voffset,
                };
                materials.push_back(Mirror{texture});
            }
        }
        else if (auto *plastic = std::get_if<ParsedPlastic>(&parsed_mat))
        {
            if (auto *color = std::get_if<Vector3>(&plastic->reflectance)) {
                ConstTexture texture = {*color};
                materials.push_back(Plastic{texture, plastic->eta});
            } else if (auto *color = std::get_if<ParsedImageTexture>(&plastic->reflectance)) {
                std::string texture_name = color->filename.string();
                int texture_id;
                auto it = textures.image3s_map.find(texture_name);
                if (it != textures.image3s_map.end()) {
                    texture_id = textures.image3s_map.at(texture_name);
                } else {
                    texture_id = textures.image3s.size();
                    textures.image3s_map.emplace(texture_name, texture_id);
                    textures.image3s.push_back(imread3(color->filename));
                }
                ImageTexture texture = {
                    texture_id,
                    color->uscale, color->vscale,
                    color->uoffset, color->voffset,
                };
                materials.push_back(Plastic{texture, plastic->eta});
            }
        }
        else if (auto *phong = std::get_if<ParsedPhong>(&parsed_mat))
        {
            if (auto *color = std::get_if<Vector3>(&phong->reflectance)) {
                ConstTexture texture = {*color};
                materials.push_back(Phong{texture, phong->exponent});
            } else if (auto *color = std::get_if<ParsedImageTexture>(&phong->reflectance)) {
                std::string texture_name = color->filename.string();
                int texture_id;
                auto it = textures.image3s_map.find(texture_name);
                if (it != textures.image3s_map.end()) {
                    texture_id = textures.image3s_map.at(texture_name);
                } else {
                    texture_id = textures.image3s.size();
                    textures.image3s_map.emplace(texture_name, texture_id);
                    textures.image3s.push_back(imread3(color->filename));
                }
                ImageTexture texture = {
                    texture_id,
                    color->uscale, color->vscale,
                    color->uoffset, color->voffset,
                };
                materials.push_back(Phong{texture, phong->exponent});
            }
        }
        else if (auto *blinn_phong = std::get_if<ParsedBlinnPhong>(&parsed_mat))
        {
            if (auto *color = std::get_if<Vector3>(&blinn_phong->reflectance)) {
                ConstTexture texture = {*color};
                materials.push_back(BlinnPhong{texture, blinn_phong->exponent});
            } else if (auto *color = std::get_if<ParsedImageTexture>(&blinn_phong->reflectance)) {
                std::string texture_name = color->filename.string();
                int texture_id;
                auto it = textures.image3s_map.find(texture_name);
                if (it != textures.image3s_map.end()) {
                    texture_id = textures.image3s_map.at(texture_name);
                } else {
                    texture_id = textures.image3s.size();
                    textures.image3s_map.emplace(texture_name, texture_id);
                    textures.image3s.push_back(imread3(color->filename));
                }
                ImageTexture texture = {
                    texture_id,
                    color->uscale, color->vscale,
                    color->uoffset, color->voffset,
                };
                materials.push_back(BlinnPhong{texture, blinn_phong->exponent});
            }
        }
        else if (auto *blinn_phong_microfacet = std::get_if<ParsedBlinnPhongMicrofacet>(&parsed_mat))
        {
            if (auto *color = std::get_if<Vector3>(&blinn_phong_microfacet->reflectance)) {
                ConstTexture texture = {*color};
                materials.push_back(BlinnPhongMicrofacet{texture, blinn_phong_microfacet->exponent});
            } else if (auto *color = std::get_if<ParsedImageTexture>(&blinn_phong_microfacet->reflectance)) {
                std::string texture_name = color->filename.string();
                int texture_id;
                auto it = textures.image3s_map.find(texture_name);
                if (it != textures.image3s_map.end()) {
                    texture_id = textures.image3s_map.at(texture_name);
                } else {
                    texture_id = textures.image3s.size();
                    textures.image3s_map.emplace(texture_name, texture_id);
                    textures.image3s.push_back(imread3(color->filename));
                }
                ImageTexture texture = {
                    texture_id,
                    color->uscale, color->vscale,
                    color->uoffset, color->voffset,
                };
                materials.push_back(BlinnPhongMicrofacet{texture, blinn_phong_microfacet->exponent});
            }
        }
        else
        {
            assert(false);
        }
    }
    for (const ParsedLight &parsed_light : scene.lights)
    {
        if (auto* point_light = std::get_if<ParsedPointLight>(&parsed_light)){
            lights.push_back(PointLight{point_light->intensity, point_light->position});
        }
    }
    // Reset shape id of area light
    for (int i = 0; i < shapes.size(); ++i) {
        int area_light_id = -1;
        Shape& shape = shapes.at(i);
        if (auto *sph = std::get_if<Sphere>(&shape)) {
            area_light_id = sph->area_light_id;
        } else if (auto *tri = std::get_if<Triangle>(&shape)) {
            area_light_id = tri->area_light_id;
        }
        if (area_light_id == -1)
            continue;
        if (auto *area_light = std::get_if<ParsedDiffuseAreaLight>(&scene.lights.at(area_light_id))) {
            int new_light_id = lights.size();
            lights.push_back(DiffuseAreaLight{i, area_light->radiance});
            if (auto *sph = std::get_if<Sphere>(&shape)) {
                sph->area_light_id = new_light_id;
            } else if (auto *tri = std::get_if<Triangle>(&shape)) {
                tri->area_light_id = new_light_id;
            }
        }
    }
}

void build_bvh(Scene& scene) {
    std::vector<BBoxWithID> bboxes(scene.shapes.size());
    for (int i = 0; i < (int)bboxes.size(); i++) {
        if (auto *sph = std::get_if<Sphere>(&scene.shapes[i])) {
            Vector3 p_min = sph->center - sph->radius;
            Vector3 p_max = sph->center + sph->radius;
            bboxes[i] = {BBox{p_min, p_max}, i};
        } else if (auto *tri = std::get_if<Triangle>(&scene.shapes[i])) {
            const TriangleMesh *mesh = &scene.meshes[tri->mesh_id];
            Vector3i index = mesh->indices[tri->face_index];
            Vector3 p0 = mesh->positions[index[0]];
            Vector3 p1 = mesh->positions[index[1]];
            Vector3 p2 = mesh->positions[index[2]];
            Vector3 p_min = min(min(p0, p1), p2);
            Vector3 p_max = max(max(p0, p1), p2);
            bboxes[i] = {BBox{p_min, p_max}, i};
        }
    }
    scene.bvh_root_id = construct_bvh(bboxes, scene.bvh_nodes);
}

__device__ std::optional<Intersection> bvh_intersect(const deviceScene& scene, const BVHNode &node, Ray ray) {
    if (node.primitive_id != -1) {
        return intersect_shape(scene, scene.shapes[node.primitive_id], ray);
    }
    const BVHNode &left = scene.bvh_nodes[node.left_node_id];
    const BVHNode &right = scene.bvh_nodes[node.right_node_id];
    std::optional<Intersection> isect_left;
    if (intersect(left.box, ray)) {
        isect_left = bvh_intersect(scene, left, ray);
        if (isect_left) {
            ray.tmax = isect_left->t;
        }
    }
    if (intersect(right.box, ray)) {
        // Since we've already set ray.tfar to the left node
        // if we still hit something on the right, it's closer
        // and we should return that.
        if (auto isect_right = bvh_intersect(scene, right, ray)) {
            return isect_right;
        }
    }
    return isect_left;
}

__device__ std::optional<Intersection> scene_intersect(const deviceScene& scene, const Ray& r){
    if(nullptr != scene.bvh_nodes){
        return bvh_intersect(scene, scene.bvh_nodes[scene.bvh_root_id], r);
    }else{
        return {};
    }
}

__device__ bool scene_occluded(const deviceScene& scene, const Ray& r){
    if(nullptr != scene.bvh_nodes){
        std::optional<Intersection> v_ = bvh_intersect(scene, scene.bvh_nodes[scene.bvh_root_id], r);
        return v_ ? true : false;
    }else{
        return false;
    }
}

__device__ Vector3 trace_ray(const deviceScene& scene, const sceneInfo& scene_info, const Ray& ray, RNGf& rng){
    Ray r = ray;
    std::optional<Intersection> v_ = scene_intersect(scene, r);
    if(!v_) return scene_info.background_color;
    Intersection v = *v_;

    Vector3 radiance = {Real(0), Real(0), Real(0)};
    Vector3 throughput = {Real(1), Real(1), Real(1)};
    for(int i = 0; i <= scene_info.options.max_depth; ++i){
        if(v.area_light_id != -1) {
            const Light& light = scene.lights[v.area_light_id];
            if (auto* l = std::get_if<DiffuseAreaLight>(&light)){
                // std::cout << throughput << std::endl;
                radiance += throughput * l->intensity;
                break;
            }
        }

        
        Vector3 dir_in = -r.dir;
        const Material& m = scene.materials[v.material_id];
        bool is_specular = false;
        if(std::holds_alternative<Plastic>(m) || std::holds_alternative<Mirror>(m))
            is_specular = true;
        
        if(scene.num_lights > 0 && !is_specular && random_double(rng) <= 0.5){
            // Sampling Light
            int light_id = sample_light(scene, rng);
            auto light = scene.lights[light_id];
            if (auto* l = std::get_if<DiffuseAreaLight>(&light)) {
                auto& light_point = sample_on_light(scene, *l, v.pos, rng);
                auto& [light_pos, light_n] = light_point;
                Real d = length(light_pos - v.pos);
                Vector3 light_dir = normalize(light_pos - v.pos);

                Real light_pdf = get_light_pdf(scene, light_id, light_point, v.pos) * (d * d) / (fmax(dot(-light_n, light_dir), Real(0)) * scene.num_lights);
                if(light_pdf <= 0){
                    // std::cout << light_pdf << "light pdf break" << std::endl;
                    break;
                }
                Real bsdf_pdf = get_bsdf_pdf(m, dir_in, light_dir, v, scene.textures);
                if(bsdf_pdf <= 0){
                    // std::cout << "bsdf pdf break" << std::endl;
                    break;
                }
                
                SampleRecord record = {};
                record.dir_out = light_dir;
                Vector3 FG = eval(m, dir_in, record, v, scene.textures);

                r = Ray{v.pos, light_dir, c_EPSILON, infinity<Real>()};
                std::optional<Intersection> v_ = scene_intersect(scene, r);
                if(!v_){
                    // std::cout << "bg break" << std::endl;
                    radiance += throughput * scene_info.background_color;
                    break;
                }
                v = *v_;
                if(v.area_light_id == -1){
                    break;
                }
                throughput *= FG / (0.5 * light_pdf + 0.5 * bsdf_pdf);
            }
        }else{
            // Sampling bsdf
            Vector3 n = dot(dir_in, v.shading_normal) < 0 ? -v.shading_normal : v.shading_normal;
            std::optional<SampleRecord> record_ = sample_bsdf(m, dir_in, v, scene.textures, rng);
            if(!record_){
                // std::cout << "record break" << std::endl;
                break;
            }
            SampleRecord& record = *record_;
            Vector3 FG = eval(m, dir_in, record, v, scene.textures);
            Vector3 dir_out = normalize(record.dir_out);
            Real bsdf_pdf = record.pdf;
            if(bsdf_pdf <= Real(0)){
                // std::cout << "pdf break" << std::endl;
                break;
            }
            r = Ray{v.pos, dir_out, c_EPSILON, infinity<Real>()};
            std::optional<Intersection> new_v_ = scene_intersect(scene, r);

            Real pdf = ((nullptr == scene.lights) || is_specular) ? bsdf_pdf : 0.5 * bsdf_pdf;

            if(!new_v_){
                // std::cout << "bg break" << std::endl;
                throughput *= FG / pdf;
                radiance += throughput * scene_info.background_color;
                break;
            }
            if(!is_specular && new_v_->area_light_id != -1){
                Vector3 &light_pos = new_v_->pos;
                Real d = length(light_pos - v.pos);
                Vector3 light_dir = normalize(light_pos - v.pos);
                Real light_pdf = get_light_pdf(scene, new_v_->area_light_id, {new_v_->pos, new_v_->geo_normal}, v.pos) * (d * d) / (fmax(dot(-new_v_->geo_normal, light_dir), Real(0)) * scene.num_lights);
                if(light_pdf <= 0){
                    // std::cout << dot(-new_v_->geo_normal, light_dir) << std::endl;
                    break;
                }
                pdf += 0.5 * light_pdf;
            }
            throughput *= FG / pdf;
            v = *new_v_;
        }
    }
    return radiance;
}