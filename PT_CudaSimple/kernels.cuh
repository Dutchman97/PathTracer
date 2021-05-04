#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Structures.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture);

__global__ void Initialize(Ray* rays, size_t rayArrayPitch, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 topRight, float4 bottomLeft);
