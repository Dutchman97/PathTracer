﻿#include "kernels.cuh"

#include <CUDA/helper_math.h>
#include <float.h>

__global__ void DrawToTexture(cudaSurfaceObject_t texture, int screenWidth, int screenHeight, float* tValues) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= screenWidth * screenHeight) return;

	uint x = i % screenWidth;
	uint y = i / screenWidth;

	float red = (tValues[i] > EPSILON && tValues[i] < FLT_MAX) ? 1.0f : 0.2f;

	// IMPORTANT: Surface functions use bytes for addressing memory; x-coordinate is in bytes.
	// Y-coordinate does not need to be multiplied as the byte offset of the corresponding y-coordinate is internally calculated.
	surf2Dwrite(make_float4(red, 0.2f, 0.2f, 1.0f), texture, x * 4 * 4, y);
}

__global__ void InitializeRng(curandStateXORWOW_t* rngStates, int count) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= count) return;

	curand_init(1337 + i, 0, 0, &rngStates[i]);
}

__global__ void InitializeRays(Ray* rays, curandStateXORWOW_t* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 bottomLeft, float4 bottomRight, float* tValues) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= screenWidth * screenHeight) return;

	uint x = i % screenWidth;
	uint y = i / screenWidth;

	float xScreen = ((float)x + curand_uniform(&rngStates[i])) / screenWidth;
	float yScreen = ((float)y + curand_uniform(&rngStates[i])) / screenHeight;

	Ray* rayPtr = &rays[i];
	rayPtr->origin = origin;
	rayPtr->direction = normalize(bottomLeft + (bottomRight - bottomLeft) * xScreen + (topLeft - bottomLeft) * yScreen);

	tValues[i] = FLT_MAX;
}

// Uses the intersection algorithm by Möller and Trumbore.
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ float RayIntersectsTriangle(Ray* rayPtr, Triangle* trianglePtr, Vertex* vertices) {
	int idx0 = trianglePtr->vertexIdx0;
	int idx1 = trianglePtr->vertexIdx1;
	int idx2 = trianglePtr->vertexIdx2;

	float4 v0 = vertices[idx0].position;
	float4 v1 = vertices[idx1].position;
	float4 v2 = vertices[idx2].position;

	float4 edge0 = v1 - v0;
	float4 edge1 = v2 - v0;

	float4 h = cross(rayPtr->direction, edge1);
	float determinant = dot(edge0, h);

#ifdef CULLING_ENABLED
	if (determinant < EPSILON) return 0.0f;
#else
	if (determinant > -EPSILON || determinant < EPSILON) return 0.0f;
#endif

	float f = 1.0f / determinant;
	float4 s = rayPtr->origin - v0;

	float u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f) return 0.0f;

	float4 q = cross(s, edge0);
	float v = f * dot(rayPtr->direction, q);
	if (v < 0.0f || u + v > 1.0f) return 0.0f;

	float t = f * dot(edge1, q);
	return t;
}

__global__ void TraverseScene(Ray* rays, int rayCount, Triangle* triangles, int triangleCount, Vertex* vertices, float* tValues) {
	uint rayIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (rayIdx >= rayCount) return;

	for (int triangleIdx = 0; triangleIdx < triangleCount; triangleIdx++) {
		float t = RayIntersectsTriangle(&rays[rayIdx], &triangles[triangleIdx], vertices);
		if (t > EPSILON && t < tValues[rayIdx]) {
			tValues[rayIdx] = t;
		}
	}
}