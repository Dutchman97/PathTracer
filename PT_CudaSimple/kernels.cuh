#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <CUDA/helper_math.h>

#include "Structures.cuh"

#define CULLING_ENABLED

__global__ void DrawToTexture(cudaSurfaceObject_t texture, int screenWidth, int screenHeight, Intersection* intersections, uint frameNumber, float4* frameBuffer);

__global__ void InitializeRng(curandStateXORWOW_t* rngStates, int count);

__global__ void InitializeRays(Ray* rays, curandStateXORWOW_t* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 bottomLeft, float4 bottomRight, Intersection* intersections, float4* frameBuffer);

inline __host__ __device__ float4 cross(float4 a, float4 b) {
	return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f);
}

inline __host__ __device__ bool operator==(const float4& a, const float4& b) {
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

__device__ Intersection RayIntersectsTriangle(Ray* rayPtr, Triangle* trianglePtr, Vertex* vertices);

__global__ void TraverseScene(Ray* rays, int rayCount, Triangle* triangles, int triangleCount, Vertex* vertices, Intersection* intersections);

inline __host__ __device__ float lengthSquared(float4 a) {
	return dot(a, a);
}

__device__ float4 GetDiffuseReflection(float4 normal, curandStateXORWOW_t* rngStatePtr);

__global__ void Intersect(Ray* rays, int rayCount, Intersection* intersections, Material* materials, curandStateXORWOW_t* rngStates, float4* frameBuffer);
