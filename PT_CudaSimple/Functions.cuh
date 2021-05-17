#pragma once

#include <cuda_runtime.h>
#include "Structures.cuh"

template<class T>
inline __device__ T* GetFromPitchedMemory(T* ptr, size_t pitch, int col, int row) {
	return (T*)((char*)ptr + row * pitch) + col;
}

inline __host__ __device__ float4 cross(float4 a, float4 b) {
	return make_float4(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x, 0.0f);
}

inline __host__ __device__ bool operator==(const float4& a, const float4& b) {
	return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline __host__ __device__ float lengthSquared(float4 a) {
	return dot(a, a);
}



// Uses the intersection algorithm by Möller and Trumbore.
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ Intersection RayIntersectsTriangle(Ray* rayPtr, Triangle* trianglePtr, Vertex* vertices) {
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
	if (determinant < EPSILON) return NO_INTERSECTION;
#else
	if (determinant > -EPSILON || determinant < EPSILON) return NO_INTERSECTION;
#endif

	float f = 1.0f / determinant;
	float4 s = rayPtr->origin - v0;

	float u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f) return NO_INTERSECTION;

	float4 q = cross(s, edge0);
	float v = f * dot(rayPtr->direction, q);
	if (v < 0.0f || u + v > 1.0f) return NO_INTERSECTION;

	float t = f * dot(edge1, q);
	return Intersection{ t, trianglePtr->materialIdx, cross(edge0, edge1) };
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
