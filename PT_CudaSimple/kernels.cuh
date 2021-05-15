#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "Structures.cuh"

#define CULLING_ENABLED
constexpr float EPSILON = 0.000001f;

__global__ void DrawToTexture(cudaSurfaceObject_t texture, int screenWidth, int screenHeight, Intersection* intersections, uint frameNumber);

__global__ void InitializeRng(curandStateXORWOW_t* rngStates, int count);

__global__ void InitializeRays(Ray* rays, curandStateXORWOW_t* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 bottomLeft, float4 bottomRight, Intersection* intersections, float4* frameBuffer);

inline __host__ __device__ float4 cross(float4 a, float4 b) {
	return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f);
}

__device__ Intersection RayIntersectsTriangle(Ray* rayPtr, Triangle* trianglePtr, Vertex* vertices);

__global__ void TraverseScene(Ray* rays, int rayCount, Triangle* triangles, int triangleCount, Vertex* vertices, Intersection* intersections);

inline __host__ __device__ float lengthSquared(float4 a) {
	return dot(a, a);
}

__device__ float4 GetDiffuseReflection(float4 normal, curandStateXORWOW_t* rngStatePtr) {
	// Generate a random vector.
	// We want to make sure this random vector is not longer than 1, because we normalize this vector,
	// the resulting vector may skew towards the corners of the unit cube because of how we obtain this random vector.
	// Also ensure this loop does execture infinitely.
	// This still causes a slight skew towards the corners, so a better way to obtain a random unit vector is necessary.
	float4 result;
	uint loopCounter = 0;
	do {
		result = make_float4(curand_uniform(rngStatePtr), curand_uniform(rngStatePtr), curand_uniform(rngStatePtr), 0.0f);
		result = result * 2.0f - 1.0f;

		loopCounter++;
	} while (lengthSquared(result) > 1.0f && loopCounter <= 10);

	// Normalize the result and ensure it's pointing in the normal vector's hemisphere half.
	result = normalize(result);
	return dot(result, normal) > 0.0f ? result : -result;
}

__global__ void Intersect(Ray* rays, int rayCount, Intersection* intersections, Material* materials, curandStateXORWOW_t* rngStates, float4* frameBuffer);
