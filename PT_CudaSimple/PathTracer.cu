#include "PathTracer.cuh"
#include <cuda_gl_interop.h>

#include "kernels.cuh"

#ifdef _DEBUG
#define CUDA_CALL(expr) _CheckCudaError(expr, #expr)
#define CUDA_GET_LAST_ERROR CUDA_CALL(cudaGetLastError())
#else
#define CUDA_CALL(expr) expr
#define CUDA_GET_LAST_ERROR
#endif
#define BLOCK_COUNT_AND_SIZE(blockSize) this->_GetBlockCount(blockSize), blockSize

PathTracer::PathTracer(const GLuint glTexture, const int pixelWidth, const int pixelHeight) : _width(pixelWidth), _height(pixelHeight), _glTexture(glTexture) {
	uint cudaDeviceCount;
	int cudaDevices[4];
	CUDA_CALL(cudaGLGetDevices(&cudaDeviceCount, cudaDevices, 4, cudaGLDeviceList::cudaGLDeviceListAll));

	std::cout << "Found " << cudaDeviceCount << " CUDA-capable devices linked to the current OpenGL context, using first device (device " << cudaDevices[0] << ")" << std::endl;

	this->_PrintDeviceInfo(cudaDevices[0]);
	CUDA_CALL(cudaSetDevice(cudaDevices[0]));

	// Allocate memory on the GPU.
	CUDA_CALL(cudaMalloc(&this->_devicePtrs.rays, pixelWidth * pixelHeight * sizeof(Ray)));
	CUDA_CALL(cudaMalloc(&this->_devicePtrs.rngStates, pixelWidth * pixelHeight * sizeof(curandStateXORWOW_t)));

	// Get block size and # blocks that maximizes occupancy.
	int minBlockCount; // Will not be used.
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minBlockCount, &this->_kernelBlockSizes.initializeRays, InitializeRays));
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minBlockCount, &this->_kernelBlockSizes.initializeRng, InitializeRng));

	// RNG setup.
	InitializeRng<<<BLOCK_COUNT_AND_SIZE(this->_kernelBlockSizes.initializeRng)>>>(this->_devicePtrs.rngStates);
	CUDA_GET_LAST_ERROR;
	CUDA_CALL(cudaDeviceSynchronize());
}

PathTracer::~PathTracer() {
	std::cout << "Terminating CUDA path tracer" << std::endl;

	if (this->_drawingVariables.drawState == DrawState::Drawing) {
		this->FinalizeDrawing();
	}

	// Free the allocated memory on the GPU.
	CUDA_CALL(cudaFree(this->_devicePtrs.rays));
	CUDA_CALL(cudaFree(this->_devicePtrs.rngStates));

	CUDA_CALL(cudaDeviceReset());
}

void PathTracer::Update() {

}

void PathTracer::BeginDrawing() {
	this->_drawingVariables.drawState = DrawState::Drawing;

	this->_MapTexture(
		this->_glTexture,
		&this->_drawingVariables.cudaTextureResource,
		&this->_drawingVariables.cudaSurface
	);

	InitializeRays<<<BLOCK_COUNT_AND_SIZE(this->_kernelBlockSizes.initializeRays)>>>(
		this->_devicePtrs.rays,
		this->_width, this->_height,
		make_float4(0.0f, 0.0f, 0.0f, 0.0f),
		make_float4(-1.0f, 1.0f, 1.0f, 0.0f),
		make_float4(1.0f, 1.0f, 1.0f, 0.0f),
		make_float4(-1.0f, -1.0f, 1.0f, 0.0f)
	);
	CUDA_GET_LAST_ERROR;
}

void PathTracer::FinalizeDrawing() {
	if (this->_drawingVariables.drawState != DrawState::Drawing) {
		return;
	}
	this->_drawingVariables.drawState = DrawState::Idle;

	CUDA_CALL(cudaDeviceSynchronize());

	this->_UnmapTexture(
		&this->_drawingVariables.cudaTextureResource,
		&this->_drawingVariables.cudaSurface
	);
}

void PathTracer::Resize(const int pixelWidth, const int pixelHeight) {

}

inline int PathTracer::_GetBlockCount(const int blockSize) const {
	int pixelCount = this->_width * this->_height;
	return (pixelCount + blockSize - 1) / blockSize; // Round up
}

inline void PathTracer::_CheckCudaError(const cudaError_t cudaStatus, const char* functionName) {
	if (cudaStatus != cudaError::cudaSuccess) {
		std::cout << "Failed to execute '" << functionName << "' (" << cudaGetErrorName(cudaStatus) << ")" << std::endl <<
			"\t" << cudaGetErrorString(cudaStatus) << std::endl;
		throw std::exception();
	}
}

void PathTracer::_MapTexture(const GLuint glTexture, cudaGraphicsResource_t* cudaResourcePtr, cudaSurfaceObject_t* cudaSurfacePtr) const {
	// Mark the GL texture for use with CUDA.
	CUDA_CALL(cudaGraphicsGLRegisterImage(cudaResourcePtr, glTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlags::cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CUDA_CALL(cudaGraphicsMapResources(1, cudaResourcePtr));

	// Create a surface object that points to the GL texture's data.
	cudaResourceDesc resourceDesc = cudaResourceDesc();
	resourceDesc.resType = cudaResourceType::cudaResourceTypeArray;
	CUDA_CALL(cudaGraphicsSubResourceGetMappedArray(&resourceDesc.res.array.array, *cudaResourcePtr, 0, 0));
	CUDA_CALL(cudaCreateSurfaceObject(cudaSurfacePtr, &resourceDesc));
}

void PathTracer::_UnmapTexture(cudaGraphicsResource_t* cudaResourcePtr, cudaSurfaceObject_t* cudaSurfacePtr) const {
	CUDA_CALL(cudaDestroySurfaceObject(*cudaSurfacePtr));
	CUDA_CALL(cudaGraphicsUnmapResources(1, cudaResourcePtr));
	CUDA_CALL(cudaGraphicsUnregisterResource(*cudaResourcePtr));
}

void PathTracer::_PrintDeviceInfo(const int device) const {
	cudaDeviceProp properties;
	CUDA_CALL(cudaGetDeviceProperties(&properties, device));

	std::printf("Using '%s'\n", properties.name);
	std::printf("\tCompute capability:      %i.%i\n", properties.major, properties.minor);
	std::printf("\tMultiprocessors:         %i\n", properties.multiProcessorCount);
	std::printf("\tWarp size:               %i\n", properties.warpSize);
	std::printf("\tConcurrent engine count: %i\n", properties.asyncEngineCount);
}
