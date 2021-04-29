#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void DrawToTexture(cudaSurfaceObject_t texture, const int width, const int height);