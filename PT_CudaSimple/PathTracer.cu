#include "PathTracer.cuh"
#include <cuda_gl_interop.h>

#include "kernels.cuh"
#include <string>

#ifdef _DEBUG
#define CUDA_CALL(expr) _CheckCudaError(expr, #expr)
#define CUDA_GET_LAST_ERROR CUDA_CALL(cudaGetLastError())
#else
#define CUDA_CALL(expr) expr
#define CUDA_GET_LAST_ERROR
#endif
#define BLOCK_COUNT_AND_SIZE(blockSize) this->_GetBlockCount(blockSize), blockSize

PathTracer::PathTracer(const GLuint glTexture, const int pixelWidth, const int pixelHeight, const CameraData* cameraData)
	: _width(pixelWidth), _height(pixelHeight), _glTexture(glTexture), _camera(cameraData, (float)pixelWidth / pixelHeight) {
	uint cudaDeviceCount;
	int cudaDevices[4];
	CUDA_CALL(cudaGLGetDevices(&cudaDeviceCount, cudaDevices, 4, cudaGLDeviceList::cudaGLDeviceListAll));

	std::cout << "Found " << cudaDeviceCount << " CUDA-capable devices linked to the current OpenGL context, using first device (device " << cudaDevices[0] << ")" << std::endl;

	this->_PrintDeviceInfo(cudaDevices[0]);
	CUDA_CALL(cudaSetDevice(cudaDevices[0]));

	// Get block size and # blocks that maximizes occupancy.
	int minBlockCount; // Will not be used.
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minBlockCount, &this->_kernelBlockSizes.initializeRays, InitializeRays));
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minBlockCount, &this->_kernelBlockSizes.initializeRng, InitializeRng));
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minBlockCount, &this->_kernelBlockSizes.traverseScene, TraverseScene));
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minBlockCount, &this->_kernelBlockSizes.intersect, Intersect));
	CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&minBlockCount, &this->_kernelBlockSizes.drawToTexture, DrawToTexture));

	this->_AllocateDrawingMemory();
	this->_InitializeRendering();




	Vertex vertices[8] {
		Vertex { make_float4(-0.5f,  0.5f, 1.0f, 1.0f) },
		Vertex { make_float4( 0.5f,  0.5f, 1.0f, 1.0f) },
		Vertex { make_float4( 0.5f, -0.5f, 1.0f, 1.0f) },
		Vertex { make_float4(-0.5f, -0.5f, 1.0f, 1.0f) },

		Vertex { make_float4(-0.2f,  0.5f, 0.5f, 1.0f) },
		Vertex { make_float4( 0.2f,  0.5f, 0.5f, 1.0f) },
		Vertex { make_float4( 0.2f,  0.5f, 0.9f, 1.0f) },
		Vertex { make_float4(-0.2f,  0.5f, 0.9f, 1.0f) },
	};
	Triangle triangles[4] {
		Triangle { 0, 1, 2, 0 },
		Triangle { 2, 3, 0, 0 },

		Triangle { 4, 5, 6, 1 },
		Triangle { 6, 7, 4, 1 },
	};
	Material materials[2] {
		Material { Material::MaterialType::DIFFUSE, make_float4(1.0f, 0.2f, 0.2f, 1.0f), 0.0f },
		Material { Material::MaterialType::EMISSIVE, make_float4(10.0f, 10.0f, 10.0f, 10.0f), 0.0f },
	};

	CUDA_CALL(cudaMalloc(&this->_devicePtrs.vertices, 8 * sizeof(Vertex)));
	CUDA_CALL(cudaMalloc(&this->_devicePtrs.triangles, 4 * sizeof(Triangle)));
	CUDA_CALL(cudaMalloc(&this->_devicePtrs.materials, 2 * sizeof(Material)));

	CUDA_CALL(cudaMemcpy(this->_devicePtrs.vertices, vertices, 8 * sizeof(Vertex), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(this->_devicePtrs.triangles, triangles, 4 * sizeof(Triangle), cudaMemcpyKind::cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(this->_devicePtrs.materials, materials, 2 * sizeof(Material), cudaMemcpyKind::cudaMemcpyHostToDevice));
}

PathTracer::~PathTracer() {
	std::cout << "Terminating CUDA path tracer" << std::endl;

	if (this->_drawingVariables.drawState == DrawState::Drawing) {
		this->FinalizeDrawing();
	}

	// Free the allocated memory on the GPU.
	CUDA_CALL(cudaFree(this->_devicePtrs.rays));
	CUDA_CALL(cudaFree(this->_devicePtrs.rngStates));
	CUDA_CALL(cudaFree(this->_devicePtrs.intersections));

	CUDA_CALL(cudaFree(this->_devicePtrs.triangles));
	CUDA_CALL(cudaFree(this->_devicePtrs.vertices));
	CUDA_CALL(cudaFree(this->_devicePtrs.materials));

	CUDA_CALL(cudaDeviceReset());
}

void PathTracer::Update(const CameraData* cameraData) {
	this->_shouldRestartRendering = this->_camera.TryUpdate(cameraData);
}

void PathTracer::BeginDrawing() {
	this->_drawingVariables.drawState = DrawState::Drawing;

	if (this->_shouldRestartRendering) {
		this->_InitializeRendering();
	}

	this->_MapTexture(
		this->_glTexture,
		&this->_drawingVariables.cudaTextureResource,
		&this->_drawingVariables.cudaSurface
	);

	InitializeRays<<<BLOCK_COUNT_AND_SIZE(this->_kernelBlockSizes.initializeRays)>>>(
		this->_devicePtrs.rays,
		this->_devicePtrs.rngStates,
		this->_width, this->_height,
		this->_camera.GetPosition(),
		this->_camera.GetTopLeft(),
		this->_camera.GetBottomLeft(),
		this->_camera.GetBottomRight(),
		this->_devicePtrs.intersections,
		this->_devicePtrs.frameBuffer
	);
	for (int i = 0; i < 10; i++) {
		TraverseScene<<<BLOCK_COUNT_AND_SIZE(this->_kernelBlockSizes.traverseScene)>>>(
			this->_devicePtrs.rays,
			this->_width * this->_height,
			this->_devicePtrs.triangles, 4,
			this->_devicePtrs.vertices,
			this->_devicePtrs.intersections
		);
		Intersect<<<BLOCK_COUNT_AND_SIZE(this->_kernelBlockSizes.intersect)>>>(
			this->_devicePtrs.rays,
			this->_width * this->_height,
			this->_devicePtrs.intersections,
			this->_devicePtrs.materials,
			this->_devicePtrs.rngStates,
			this->_devicePtrs.frameBuffer
		);
	}
	DrawToTexture<<<BLOCK_COUNT_AND_SIZE(this->_kernelBlockSizes.drawToTexture)>>>(
		this->_drawingVariables.cudaSurface,
		this->_devicePtrs.frameBuffer,
		this->_width, this->_height,
		this->_frameNumber
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
	this->_frameNumber++;
}

void PathTracer::Resize(const int pixelWidth, const int pixelHeight) {
	this->_width = pixelWidth;
	this->_height = pixelHeight;

	this->_camera.Resize(pixelWidth, pixelHeight);

	this->_AllocateDrawingMemory();
	this->_InitializeRendering();
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

void PathTracer::_AllocateDrawingMemory() {
	if (this->_devicePtrs.rays != nullptr) {
		CUDA_CALL(cudaFree(this->_devicePtrs.rays));
	}
	if (this->_devicePtrs.rngStates != nullptr) {
		CUDA_CALL(cudaFree(this->_devicePtrs.rngStates));
	}
	if (this->_devicePtrs.intersections != nullptr) {
		CUDA_CALL(cudaFree(this->_devicePtrs.intersections));
	}
	if (this->_devicePtrs.frameBuffer != nullptr) {
		CUDA_CALL(cudaFree(this->_devicePtrs.frameBuffer));
	}

	CUDA_CALL(cudaMalloc(&this->_devicePtrs.rays, this->_width * this->_height * sizeof(Ray)));
	CUDA_CALL(cudaMalloc(&this->_devicePtrs.rngStates, this->_width * this->_height * sizeof(curandStateXORWOW_t)));
	CUDA_CALL(cudaMalloc(&this->_devicePtrs.intersections, this->_width * this->_height * sizeof(Intersection)));
	CUDA_CALL(cudaMalloc(&this->_devicePtrs.frameBuffer, this->_width * this->_height * sizeof(float4)));
}

void PathTracer::_InitializeRendering() {
	this->_shouldRestartRendering = false;
	this->_frameNumber = 0;

	InitializeRng<<<BLOCK_COUNT_AND_SIZE(this->_kernelBlockSizes.initializeRng)>>>(
		this->_devicePtrs.rngStates,
		this->_width * this->_height
	);
	CUDA_GET_LAST_ERROR;
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
