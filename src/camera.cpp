#include "camera.h"

Camera from_parsed_camera(const ParsedCamera &pc) {
    Camera c;
    c.lookat = pc.lookat;
    c.lookfrom = pc.lookfrom;
    c.up = pc.up;
    c.vfov = pc.vfov;
    return c;
}