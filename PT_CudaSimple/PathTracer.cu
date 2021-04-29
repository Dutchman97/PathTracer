#include "PathTracer.cuh"
#include <cuda_gl_interop.h>
#include <iostream>

PathTracer::PathTracer(const GLuint glTexture, const int pixelWidth, const int pixelHeight) : _width(pixelWidth), _height(pixelHeight) {
	cudaError_t cudaStatus;

	uint cudaDeviceCount;
	int cudaDevices[4];
	cudaStatus = cudaGLGetDevices(&cudaDeviceCount, cudaDevices, 4, cudaGLDeviceList::cudaGLDeviceListAll);
	_CheckCudaError(cudaStatus, "cudaGLGetDevices");

	std::cout << "Found " << cudaDeviceCount << " CUDA-capable devices, using first device (device " << cudaDevices[0] << ")" << std::endl;

	cudaStatus = cudaSetDevice(cudaDevices[0]);
	_CheckCudaError(cudaStatus, "cudaSetDevice");
	
	cudaStatus = cudaGraphicsGLRegisterImage(&this->_cudaTexture, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore);
	_CheckCudaError(cudaStatus, "cudaGraphicsGLRegisterImage");
}

PathTracer::~PathTracer() {
	std::cout << "Resetting CUDA device" << std::endl;
	cudaError_t cudaStatus = cudaDeviceReset();
	_CheckCudaError(cudaStatus, "cudaDeviceReset");
}

void PathTracer::Update() {

}

void PathTracer::Draw() {

}

void PathTracer::Resize(const int pixelWidth, const int pixelHeight) {

}

void PathTracer::_CheckCudaError(const cudaError_t cudaStatus, const char* functionName) {
	if (cudaStatus != cudaError::cudaSuccess) {
		std::cout << "Failed to execute '" << functionName << "' (error " << cudaStatus << ")" << std::endl;
		throw std::exception();
	}
}
