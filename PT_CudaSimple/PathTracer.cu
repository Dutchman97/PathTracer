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

	std::cout << "Found " << cudaDeviceCount << " CUDA-capable devices, using first device (device " << cudaDevices[0] << ")" << std::endl;

	cudaStatus = cudaSetDevice(cudaDevices[0]);
	_CheckCudaError(cudaStatus, "cudaSetDevice");
}

PathTracer::~PathTracer() {
	std::cout << "Resetting CUDA device" << std::endl;
	cudaError_t cudaStatus = cudaDeviceReset();
	_CheckCudaError(cudaStatus, "cudaDeviceReset");
}

void PathTracer::Update() {

}

void PathTracer::Draw() {
	cudaError_t cudaStatus;
	cudaStatus = cudaGraphicsGLRegisterImage(&this->_cudaTexture, this->_glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore);
	_CheckCudaError(cudaStatus, "cudaGraphicsGLRegisterImage");

	cudaStatus = cudaGraphicsMapResources(1, &this->_cudaTexture);
	_CheckCudaError(cudaStatus, "cudaGraphicsMapResources");

	cudaArray_t abcdef;
	cudaStatus = cudaGraphicsSubResourceGetMappedArray(&abcdef, this->_cudaTexture, 0, 0);
	_CheckCudaError(cudaStatus, "cudaGraphicsResourceGetMappedPointer");

	cudaSurfaceObject_t surface;
	cudaResourceDesc resourceDesc = cudaResourceDesc();
	resourceDesc.resType = cudaResourceType::cudaResourceTypeArray;
	resourceDesc.res.array.array = abcdef;
	cudaStatus = cudaCreateSurfaceObject(&surface, &resourceDesc);
	_CheckCudaError(cudaStatus, "cudaCreateSurfaceObject");

	dim3 threadsPerBlock(16, 16);
	DrawToTexture<<<1, threadsPerBlock>>>(surface);

	cudaStatus = cudaGetLastError();
	_CheckCudaError(cudaStatus, "cudaGetLastError");

	cudaStatus = cudaDeviceSynchronize();
	_CheckCudaError(cudaStatus, "cudaDeviceSynchronize");

	cudaStatus = cudaDestroySurfaceObject(surface);
	_CheckCudaError(cudaStatus, "cudaDestroySurfaceObject");

	cudaStatus = cudaGraphicsUnmapResources(1, &this->_cudaTexture);
	_CheckCudaError(cudaStatus, "cudaGraphicsUnmapResources");

	cudaStatus = cudaGraphicsUnregisterResource(this->_cudaTexture);
	_CheckCudaError(cudaStatus, "cudaGraphicsUnregisterResource");
}

void PathTracer::Resize(const int pixelWidth, const int pixelHeight) {

}

inline void PathTracer::_CheckCudaError(const cudaError_t cudaStatus, const char* functionName) {
	if (cudaStatus != cudaError::cudaSuccess) {
		std::cout << "Failed to execute '" << functionName << "' (error " << cudaStatus << ")" << std::endl;
		throw std::exception();
	}
}
