#include "PathTracer.cuh"
#include <cuda_gl_interop.h>
#include <iostream>
#include <curand_kernel.h>

#include "kernels.cuh"

constexpr int TEST_WIDTH = 8, TEST_HEIGHT = 8;

PathTracer::PathTracer(const GLuint glTexture, const int pixelWidth, const int pixelHeight) : _width(pixelWidth), _height(pixelHeight), _glTexture(glTexture) {
	cudaError_t cudaStatus;
	curandStatus_t curandStatus;

	uint cudaDeviceCount;
	int cudaDevices[4];
	cudaStatus = cudaGLGetDevices(&cudaDeviceCount, cudaDevices, 4, cudaGLDeviceList::cudaGLDeviceListAll);
	_CheckCudaError(cudaStatus, "cudaGLGetDevices");

	std::cout << "Found " << cudaDeviceCount << " CUDA-capable devices linked to the current OpenGL context, using first device (device " << cudaDevices[0] << ")" << std::endl;

	this->_PrintDeviceInfo(cudaDevices[0]);

	cudaStatus = cudaSetDevice(cudaDevices[0]);
	_CheckCudaError(cudaStatus, "cudaSetDevice");

	cudaStatus = cudaMallocPitch(&this->_drawingVariables.devicePtrs.rays, &this->_drawingVariables.devicePtrs.rayArrayPitch, pixelWidth * sizeof(Ray), pixelHeight);
	_CheckCudaError(cudaStatus, "cudaMallocPitch");

	curandStateXORWOW_t* rngStates;
	size_t rngStatesPitch;
	cudaStatus = cudaMallocPitch(&rngStates, &rngStatesPitch, pixelWidth * sizeof(curandStateXORWOW_t), pixelHeight);
	_CheckCudaError(cudaStatus, "cudaMallocPitch");

	dim3 threadsPerBlock(TEST_WIDTH, TEST_HEIGHT);
	InitializeRng<<<1, threadsPerBlock>>>(rngStates, rngStatesPitch);

	float* rngValuesDevice;
	cudaStatus = cudaMalloc(&rngValuesDevice, TEST_WIDTH * TEST_HEIGHT * sizeof(float));
	_CheckCudaError(cudaStatus, "cudaMalloc");

	TestRng<<<1, threadsPerBlock>>>(rngStates, rngStatesPitch, rngValuesDevice);

	cudaStatus = cudaDeviceSynchronize();
	_CheckCudaError(cudaStatus, "cudaDeviceSynchronize");

	float* rngValues = (float*)calloc(TEST_WIDTH * TEST_HEIGHT, sizeof(float));
	cudaStatus = cudaMemcpy(rngValues, rngValuesDevice, sizeof(float) * TEST_WIDTH * TEST_HEIGHT, cudaMemcpyKind::cudaMemcpyDeviceToHost);
	_CheckCudaError(cudaStatus, "cudaMemcpy");

	for (int i = 0; i < TEST_WIDTH * TEST_HEIGHT; i++) {
		std::cout << rngValues[i] << std::endl;
	}

	cudaStatus = cudaFree(rngValuesDevice);
	_CheckCudaError(cudaStatus, "cudaFree");
	free(rngValues);

	//curandGenerator_t curandGenerator;
	//curandStatus = curandCreateGenerator(&curandGenerator, curandRngType::CURAND_RNG_PSEUDO_XORWOW);
	//_CheckCurandError(curandStatus, "curandCreateGenerator");

	//curandStatus = curandSetPseudoRandomGeneratorSeed(curandGenerator, 1337);
	//_CheckCurandError(curandStatus, "curandSetPseudoRandomGeneratorSeed");


	//curandStatus = curandDestroyGenerator(curandGenerator);
	//_CheckCurandError(curandStatus, "curandDestroyGenerator");
}

PathTracer::~PathTracer() {
	cudaError_t cudaStatus;
	std::cout << "Terminating CUDA path tracer" << std::endl;

	if (this->_drawingVariables.drawState == DrawState::Drawing) {
		this->FinalizeDrawing();
	}

	cudaStatus = cudaFree(this->_drawingVariables.devicePtrs.rays);
	_CheckCudaError(cudaStatus, "cudaFree");

	cudaStatus = cudaDeviceReset();
	_CheckCudaError(cudaStatus, "cudaDeviceReset");
}

void PathTracer::Update() {

}

void PathTracer::BeginDrawing() {
	cudaError_t cudaStatus;
	this->_drawingVariables.drawState = DrawState::Drawing;

	this->_MapTexture(
		this->_glTexture,
		&this->_drawingVariables.cudaTextureResource,
		&this->_drawingVariables.cudaSurface
	);

	dim3 threadsPerBlock(TEST_WIDTH, TEST_HEIGHT);
	//DrawToTexture<<<1, threadsPerBlock>>>(this->_drawingVariables.cudaSurface);
	InitializeRays<<<1, threadsPerBlock>>>(
		this->_drawingVariables.devicePtrs.rays, this->_drawingVariables.devicePtrs.rayArrayPitch,
		TEST_WIDTH, TEST_HEIGHT,
		//this->_width, this->_height,
		make_float4(0.0f, 0.0f, 0.0f, 0.0f),
		make_float4(-1.0f, 1.0f, 1.0f, 0.0f),
		make_float4(1.0f, 1.0f, 1.0f, 0.0f),
		make_float4(-1.0f, -1.0f, 1.0f, 0.0f)
	);

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

inline void PathTracer::_CheckCurandError(const curandStatus_t curandStatus, const char* functionName) {
	if (curandStatus != curandStatus_t::CURAND_STATUS_SUCCESS) {
		std::string errorName;
		switch (curandStatus) {
		case CURAND_STATUS_VERSION_MISMATCH: ///< Header file and linked library version do not match
			errorName = "CURAND_STATUS_VERSION_MISMATCH";
			break;
		case CURAND_STATUS_NOT_INITIALIZED: ///< Generator not initialized
			errorName = "CURAND_STATUS_NOT_INITIALIZED";
			break;
		case CURAND_STATUS_ALLOCATION_FAILED: ///< Memory allocation failed
			errorName = "CURAND_STATUS_ALLOCATION_FAILED";
			break;
		case CURAND_STATUS_TYPE_ERROR: ///< Generator is wrong type
			errorName = "CURAND_STATUS_TYPE_ERROR";
			break;
		case CURAND_STATUS_OUT_OF_RANGE: ///< Argument out of range
			errorName = "CURAND_STATUS_OUT_OF_RANGE";
			break;
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE: ///< Length requested is not a multple of dimension
			errorName = "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
			break;
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: ///< GPU does not have double precision required by MRG32k3a
			errorName = "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
			break;
		case CURAND_STATUS_LAUNCH_FAILURE: ///< Kernel launch failure
			errorName = "CURAND_STATUS_LAUNCH_FAILURE";
			break;
		case CURAND_STATUS_PREEXISTING_FAILURE: ///< Preexisting failure on library entry
			errorName = "CURAND_STATUS_PREEXISTING_FAILURE";
			break;
		case CURAND_STATUS_INITIALIZATION_FAILED: ///< Initialization of CUDA failed
			errorName = "CURAND_STATUS_INITIALIZATION_FAILED";
			break;
		case CURAND_STATUS_ARCH_MISMATCH: ///< Architecture mismatch, GPU does not support requested feature
			errorName = "CURAND_STATUS_ARCH_MISMATCH";
			break;
		case CURAND_STATUS_INTERNAL_ERROR: ///< Internal library error
			errorName = "CURAND_STATUS_INTERNAL_ERROR";
			break;
		default:
			errorName = "Unknown error";
			break;
		}
		std::cout << "Failed to execute '" << functionName << "' (" << errorName.c_str() << ")" << std::endl;
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
