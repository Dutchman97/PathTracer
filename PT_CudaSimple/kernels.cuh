#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "Structures.cuh"

#define CULLING_ENABLED
constexpr float EPSILON = 0.000001f;

__global__ void DrawToTexture(cudaSurfaceObject_t texture, int screenWidth, int screenHeight, float* tValues);

__global__ void InitializeRng(curandStateXORWOW_t* rngStates, int count);

__global__ void InitializeRays(Ray* rays, curandStateXORWOW_t* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 bottomLeft, float4 bottomRight, float* tValues);

inline __host__ __device__ float4 cross(float4 a, float4 b) {
	return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f);
}

__device__ float RayIntersectsTriangle(Ray* rayPtr, Triangle* trianglePtr, Vertex* vertices);

__global__ void TraverseScene(Ray* rays, int rayCount, Triangle* triangles, int triangleCount, Vertex* vertices, float* tValues);
