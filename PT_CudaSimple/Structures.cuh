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
	uint1 vertexIdx0;
	uint1 vertexIdx1;
	uint1 vertexIdx2;
};

struct Camera {
	float4 position;
	float4 topLeft;
};
