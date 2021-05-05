#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "Structures.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture);

__global__ void InitializeRng(curandStateXORWOW_t* rngStates, size_t rngStatesPitch);

__global__ void TestRng(curandStateXORWOW_t* rngStates, size_t rngStatesPitch, float* output);

__global__ void InitializeRays(Ray* rays, size_t rayArrayPitch, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 topRight, float4 bottomLeft);
