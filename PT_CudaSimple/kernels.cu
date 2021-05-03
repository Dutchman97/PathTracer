#include "kernels.cuh"

#include <CUDA/helper_math.h>

__global__ void DrawToTexture(cudaSurfaceObject_t texture) {
	uint x = threadIdx.x;
	uint y = threadIdx.y;

	// IMPORTANT: Surface functions use bytes for addressing memory; x-coordinate is in bytes.
	// Y-coordinate does not need to be multiplied as the byte offset of the corresponding y-coordinate is internally calculated.
	surf2Dwrite(make_float4(1.0f, 0.2f, 0.2f, 1.0f), texture, x * 4 * 4, y);
}

__global__ void Initialize(Ray* rays, size_t rayArrayPitch, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 topRight, float4 bottomLeft) {
	uint x = threadIdx.x;
	uint y = threadIdx.y;

	if (x >= screenWidth || y >= screenHeight) return;

	float xScreen = (float)x / screenWidth;
	float yScreen = (float)y / screenHeight;

	Ray* rayPtr = GetFromPitchedMemory(rays, rayArrayPitch, x, y);
	rayPtr->origin = origin;
	rayPtr->direction = topLeft + (topRight - topLeft) * xScreen + (bottomLeft - topLeft) * yScreen;
}
