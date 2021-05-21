#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <CUDA/helper_math.h>

#include "Structures.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture, float4* frameBuffer, int screenWidth, int screenHeight, uint frameNumber);

__global__ void ResetCompactionArray(CompactionArray compactionArray);

__global__ void InitializeRng(RngState* rngStates, int count);

__global__ void InitializeRays(Ray* rays, RngState* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 bottomLeft, float4 bottomRight, Intersection* intersections, float4* stepBuffer, float4* frameBuffer, CompactionArray traverseSceneCompaction);

__global__ void TraverseScene(Ray* rays, Triangle* triangles, int triangleCount, Vertex* vertices, Intersection* intersections, CompactionArray intersectCompaction, CompactionArray traverseSceneCompaction);

__global__ void Intersect(Ray* rays, Intersection* intersections, Material* materials, RngState* rngStates, float4* stepBuffer, float4* frameBuffer, CompactionArray intersectCompaction, CompactionArray traverseSceneCompaction);
