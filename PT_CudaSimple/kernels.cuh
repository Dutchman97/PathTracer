#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "Structures.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture);

__global__ void InitializeRng(curandStateXORWOW_t* rngStates);

__global__ void TestRng(curandStateXORWOW_t* rngStates, float* output);

__global__ void InitializeRays(Ray* rays, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 topRight, float4 bottomLeft);
