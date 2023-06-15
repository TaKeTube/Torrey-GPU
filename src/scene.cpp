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