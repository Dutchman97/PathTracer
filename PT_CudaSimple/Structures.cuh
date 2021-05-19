#pragma once

#include <cuda_runtime.h>
#include <float.h>

#define CULLING_ENABLED

constexpr float EPSILON = 0.000001f;

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

	inline __host__ __device__ bool Hit() const {
		return t > EPSILON && t < FLT_MAX;
	}
};

constexpr __device__ float4 ZERO_VECTOR { 0.0f, 0.0f, 0.0f, 0.0f };
constexpr __device__ Intersection NO_INTERSECTION { FLT_MAX, 0, ZERO_VECTOR };

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

struct CompactionArray {
	uint* data = nullptr;

	inline __device__ void Add(uint value) {
		uint idx = atomicInc(data, 1u << 31);
		data[idx + 1] = value;
	}

	inline __device__ uint Get(uint idx) {
		return data[idx + 1];
	}

	inline __device__ void Reset() {
		data[0] = 0;
	}

	inline __device__ uint GetCount() {
		return data[0];
	}
};
