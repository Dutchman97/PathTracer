#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <CUDA/helper_math.h>

#include "Structures.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture, float4* frameBuffer, int screenWidth, int screenHeight, uint frameNumber);

__global__ void InitializeRng(curandStateXORWOW_t* rngStates, int count);

__global__ void InitializeRays(Ray* rays, curandStateXORWOW_t* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 bottomLeft, float4 bottomRight, Intersection* intersections, float4* frameBuffer);

__global__ void TraverseScene(Ray* rays, int rayCount, Triangle* triangles, int triangleCount, Vertex* vertices, Intersection* intersections);

__global__ void Intersect(Ray* rays, int rayCount, Intersection* intersections, Material* materials, curandStateXORWOW_t* rngStates, float4* frameBuffer);
