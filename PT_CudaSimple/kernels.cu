#include "kernels.cuh"

#include <CUDA/helper_math.h>

__global__ void DrawToTexture(cudaSurfaceObject_t texture) {
	uint x = threadIdx.x;
	uint y = threadIdx.y;

	// IMPORTANT: Surface functions use bytes for addressing memory; x-coordinate is in bytes.
	// Y-coordinate does not need to be multiplied as the byte offset of the corresponding y-coordinate is internally calculated.
	surf2Dwrite(make_float4(1.0f, 0.2f, 0.2f, 1.0f), texture, x * 4 * 4, y);
}

__global__ void InitializeRng(curandStateXORWOW_t* rngStates) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;

	curand_init(1337 + i, 0, 0, &rngStates[i]);
}

__global__ void InitializeRays(Ray* rays, curandStateXORWOW_t* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 topRight, float4 bottomLeft) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	uint x = i % screenWidth;
	uint y = i / screenWidth;

	if (x >= screenWidth || y >= screenHeight) return;

	float xScreen = ((float)x + curand_uniform(&rngStates[i])) / screenWidth;
	float yScreen = ((float)y + curand_uniform(&rngStates[i])) / screenHeight;

	Ray* rayPtr = &rays[i];
	rayPtr->origin = origin;
	rayPtr->direction = normalize(topLeft + (topRight - topLeft) * xScreen + (bottomLeft - topLeft) * yScreen);
}
