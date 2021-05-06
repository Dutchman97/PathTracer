#pragma once

#include <IPathTracer.h>
#include <glad/glad.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

#include "Structures.cuh"

class PathTracer : public IPathTracer {
	// Properties
private:
	cudaGraphicsResource_t _cudaTexture;
	int _width, _height;
	GLuint _glTexture;

	enum DrawState { Idle, Drawing };
	struct DrawingVariables {
		DrawState drawState = DrawState::Idle;
		cudaGraphicsResource_t cudaTextureResource;
		cudaSurfaceObject_t cudaSurface;
	} _drawingVariables;

	struct DevicePtrs {
		Ray* rays;
		curandStateXORWOW_t* rngStates;
	} _devicePtrs;

	struct KernelBlockSizes {
		int initializeRays;
		int initializeRng;
	} _kernelBlockSizes;

	// Methods
public:
	PathTracer(const GLuint glTexture, const int pixelWidth, const int pixelHeight);
	void Update();
	void BeginDrawing();
	void FinalizeDrawing();
	void Resize(const int pixelWidth, const int pixelHeight);
	~PathTracer();
private:
	inline static void _CheckCudaError(const cudaError_t cudaStatus, const char* functionName);
	inline int _GetBlockCount(const int blockSize) const;
	void _MapTexture(const GLuint glTexture, cudaGraphicsResource_t* cudaResourcePtr, cudaSurfaceObject_t* cudaSurfacePtr) const;
	void _UnmapTexture(cudaGraphicsResource_t* cudaResourcePtr, cudaSurfaceObject_t* cudaSurfacePtr) const;
	void _PrintDeviceInfo(const int device) const;
};

typedef unsigned int uint;

extern "C" __declspec(dllexport) IPathTracer* Create(const unsigned int glTexture, const int pixelWidth, const int pixelHeight) {
	gladLoadGL();
	return new PathTracer(glTexture, pixelWidth, pixelHeight);
}

extern "C" __declspec(dllexport) void Destroy(IPathTracer* pathTracer) {
	delete (PathTracer*)pathTracer;
}

