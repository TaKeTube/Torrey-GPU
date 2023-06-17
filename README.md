# Torrey-GPU

A GPU version of CSE 168 Path Tracer Torrey. Except the scene parser and the Vector classes, this renderer is implemented from scratch.

![room](/final_scene/room.png?raw=true)

Original design scene. Using assets from https://polyhaven.com/.

**Tested Environment:**

- Windows 10
- VS 2019

- CUDA 11.3
- CMake 3.26.4

**Build Requirements:**

- CUDA 11.x
- CMake

**Key Features**

- Bounding volume hierarchies
- Textures
- Monte Carlo path tracing
- Microfacet BRDFs
- Multiple importance sampling
- CUDA acceleration