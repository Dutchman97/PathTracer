#include "PathTracer.cuh"
#include <cuda_gl_interop.h>
#include <iostream>

#include "kernels.cuh"

PathTracer::PathTracer(const GLuint glTexture, const int pixelWidth, const int pixelHeight) : _width(pixelWidth), _height(pixelHeight), _glTexture(glTexture) {
	cudaError_t cudaStatus;

	uint cudaDeviceCount;
	int cudaDevices[4];
	cudaStatus = cudaGLGetDevices(&cudaDeviceCount, cudaDevices, 4, cudaGLDeviceList::cudaGLDeviceListAll);
	_CheckCudaError(cudaStatus, "cudaGLGetDevices");

	std::cout << "Found " << cudaDeviceCount << " CUDA-capable devices linked to the current OpenGL context, using first device (device " << cudaDevices[0] << ")" << std::endl;

	this->_PrintDeviceInfo(cudaDevices[0]);

	cudaStatus = cudaSetDevice(cudaDevices[0]);
	_CheckCudaError(cudaStatus, "cudaSetDevice");
}

PathTracer::~PathTracer() {
	std::cout << "Terminating CUDA path tracer" << std::endl;

	if (this->_drawingVariables.drawState == DrawState::Drawing) {
		this->FinalizeDrawing();
	}

	cudaError_t cudaStatus = cudaDeviceReset();
	_CheckCudaError(cudaStatus, "cudaDeviceReset");
}

void PathTracer::Update() {

}

void PathTracer::BeginDrawing() {
	cudaError_t cudaStatus;
	this->_drawingVariables = DrawingVariables();
	this->_drawingVariables.drawState = DrawState::Drawing;

	this->_MapTexture(
		this->_glTexture,
		&this->_drawingVariables.cudaTextureResource,
		&this->_drawingVariables.cudaSurface
	);

	dim3 threadsPerBlock(16, 16);
	DrawToTexture<<<1, threadsPerBlock>>>(this->_drawingVariables.cudaSurface);

	cudaStatus = cudaGetLastError();
	_CheckCudaError(cudaStatus, "cudaGetLastError");
}

void PathTracer::FinalizeDrawing() {
	if (this->_drawingVariables.drawState != DrawState::Drawing) {
		return;
	}
	this->_drawingVariables.drawState = DrawState::Idle;

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceSynchronize();
	_CheckCudaError(cudaStatus, "cudaDeviceSynchronize");

	this->_UnmapTexture(
		&this->_drawingVariables.cudaTextureResource,
		&this->_drawingVariables.cudaSurface
	);
}

void PathTracer::Resize(const int pixelWidth, const int pixelHeight) {

}

inline void PathTracer::_CheckCudaError(const cudaError_t cudaStatus, const char* functionName) {
	if (cudaStatus != cudaError::cudaSuccess) {
		std::cout << "Failed to execute '" << functionName << "' (" << cudaGetErrorName(cudaStatus) << ")" << std::endl <<
			"\t" << cudaGetErrorString(cudaStatus) << std::endl;
		throw std::exception();
	}
}

void PathTracer::_MapTexture(const GLuint glTexture, cudaGraphicsResource_t* cudaResourcePtr, cudaSurfaceObject_t* cudaSurfacePtr) const {
	cudaError_t cudaStatus;
	cudaStatus = cudaGraphicsGLRegisterImage(cudaResourcePtr, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore);
	_CheckCudaError(cudaStatus, "cudaGraphicsGLRegisterImage");

	cudaStatus = cudaGraphicsMapResources(1, cudaResourcePtr);
	_CheckCudaError(cudaStatus, "cudaGraphicsMapResources");

	cudaArray_t abcdef;
	cudaStatus = cudaGraphicsSubResourceGetMappedArray(&abcdef, *cudaResourcePtr, 0, 0);
	_CheckCudaError(cudaStatus, "cudaGraphicsResourceGetMappedPointer");

	cudaResourceDesc resourceDesc = cudaResourceDesc();
	resourceDesc.resType = cudaResourceType::cudaResourceTypeArray;
	resourceDesc.res.array.array = abcdef;
	cudaStatus = cudaCreateSurfaceObject(cudaSurfacePtr, &resourceDesc);
	_CheckCudaError(cudaStatus, "cudaCreateSurfaceObject");
}

void PathTracer::_UnmapTexture(cudaGraphicsResource_t* cudaResourcePtr, cudaSurfaceObject_t* cudaSurfacePtr) const {
	cudaError_t cudaStatus;
	cudaStatus = cudaDestroySurfaceObject(*cudaSurfacePtr);
	_CheckCudaError(cudaStatus, "cudaDestroySurfaceObject");

	cudaStatus = cudaGraphicsUnmapResources(1, cudaResourcePtr);
	_CheckCudaError(cudaStatus, "cudaGraphicsUnmapResources");

	cudaStatus = cudaGraphicsUnregisterResource(*cudaResourcePtr);
	_CheckCudaError(cudaStatus, "cudaGraphicsUnregisterResource");
}

void PathTracer::_PrintDeviceInfo(const int device) const {
	cudaError_t cudaStatus;
	cudaDeviceProp properties;
	cudaStatus = cudaGetDeviceProperties(&properties, device);
	_CheckCudaError(cudaStatus, "cudaGetDeviceProperties");

	std::printf("Using '%s'\n", properties.name);
	std::printf("\tCompute capability:      %i.%i\n", properties.major, properties.minor);
	std::printf("\tMultiprocessors:         %i\n", properties.multiProcessorCount);
	std::printf("\tWarp size:               %i\n", properties.warpSize);
	std::printf("\tConcurrent engine count: %i\n", properties.asyncEngineCount);
}
