﻿#pragma once

#include <cuda_runtime.h>
#include <float.h>

template<class T>
inline __device__ T* GetFromPitchedMemory(T* ptr, size_t pitch, int col, int row) {
	return (T*)((char*)ptr + row * pitch) + col;
}

typedef unsigned int uint;

struct Ray {
	float4 origin;
	float4 direction;
};

// Another array of structures, terrible for data locality.
// When all kernels are implemented, check if turning this into a structure of arrays is worth it.
struct Intersection {
	float t;
	uint materialIdx;
	float4 normal;

	inline const bool Hit() const {
		return t > EPSILON && t < FLT_MAX;
	}
};

constexpr float4 ZERO_VECTOR { 0.0f, 0.0f, 0.0f, 0.0f };
constexpr Intersection NO_INTERSECTION { FLT_MAX, 0, ZERO_VECTOR };

struct Vertex {
	float4 position;
};

// Seperate material index from vertex indices to improve data locality? A Triangle array and a TriangleMaterial array?
struct Triangle {
	uint vertexIdx0;
	uint vertexIdx1;
	uint vertexIdx2;

	uint materialIdx;
};

struct Material {
	enum MaterialType : uint {
		DIFFUSE, REFLECTIVE, EMISSIVE,
	} type;

	float4 color;

	union {
		struct {
			float reflectivity;
		} reflective;
		// Add metadata for other materials here when they're added.
	};
};
