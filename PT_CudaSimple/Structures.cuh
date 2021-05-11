#pragma once

#include <cuda_runtime.h>

template<class T>
inline __device__ T* GetFromPitchedMemory(T* ptr, size_t pitch, int col, int row) {
	return (T*)((char*)ptr + row * pitch) + col;
}

typedef unsigned int uint;

struct Ray {
	float4 origin;
	float4 direction;
};

struct Vertex {
	float4 position;
};

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
