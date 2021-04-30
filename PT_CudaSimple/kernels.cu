#include "kernels.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture) {
	unsigned int x = threadIdx.x;
	unsigned int y = threadIdx.y;

	// IMPORTANT: Surface functions use bytes for addressing memory; x-coordinate is in bytes.
	// Y-coordinate does not need to be multiplied as the byte offset of the corresponding y-coordinate is internally calculated.
	surf2Dwrite(make_float4(1.0f, 0.2f, 0.2f, 1.0f), texture, x * 4 * 4, y);
}