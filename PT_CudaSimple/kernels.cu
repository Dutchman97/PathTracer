#include "kernels.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture, const int width, const int height) {
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	surf2Dwrite(make_float4(1.0f, 0.2f, 0.2f, 1.0f), texture, x, y);
}