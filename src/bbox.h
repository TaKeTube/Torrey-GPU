#pragma once
#include "ray.cuh"

struct BBox {
    Vector3 p_min = Vector3{ infinity<Real>(),
                            infinity<Real>(),
                            infinity<Real>() };
    Vector3 p_max = Vector3{ -infinity<Real>(),
                            -infinity<Real>(),
                            -infinity<Real>() };
};

struct BBoxWithID {
    BBox box;
    int id;
};

inline bool intersect(const BBox& bbox, Ray r) {
    Real t_min = r.tmin;
    Real t_max = r.tmax;
    for (int a = 0; a < 3; a++) {
        auto t0 = fmin((bbox.p_min[a] - r.origin[a]) / r.dir[a],
            (bbox.p_max[a] - r.origin[a]) / r.dir[a]);
        auto t1 = fmax((bbox.p_min[a] - r.origin[a]) / r.dir[a],
            (bbox.p_max[a] - r.origin[a]) / r.dir[a]);
        t_min = fmax(t0, t_min);
        t_max = fmin(t1, t_max);
        if (t_max < t_min)
            return false;
    }
    return true;
}

inline int largest_axis(const BBox& box) {
    Vector3 extent = box.p_max - box.p_min;
    if (extent.x > extent.y && extent.x > extent.z) {
        return 0;
    }
    else if (extent.y > extent.x && extent.y > extent.z) {
        return 1;
    }
    else { // z is the largest
        return 2;
    }
}

inline BBox merge(const BBox& box1, const BBox& box2) {
    Vector3 p_min = Vector3{
        std::min(box1.p_min.x, box2.p_min.x),
        std::min(box1.p_min.y, box2.p_min.y),
        std::min(box1.p_min.z, box2.p_min.z) };
    Vector3 p_max = Vector3{
        std::max(box1.p_max.x, box2.p_max.x),
        std::max(box1.p_max.y, box2.p_max.y),
        std::max(box1.p_max.z, box2.p_max.z) };
    return BBox{ p_min, p_max };
}