#pragma once

#include "torrey.cuh"
#include "matrix.h"
#include "parse_scene.h"
#include <filesystem>

/// Parse Stanford PLY files.
ParsedTriangleMesh parse_ply(const fs::path &filename, const Matrix4x4 &to_world);
