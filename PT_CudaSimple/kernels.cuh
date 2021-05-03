#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Structures.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture);