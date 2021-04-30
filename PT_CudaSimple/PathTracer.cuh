#pragma once

#include <IPathTracer.h>
#include <glad/glad.h>
#include <cuda_runtime.h>

class PathTracer : public IPathTracer {
	// Properties
private:
	cudaGraphicsResource_t _cudaTexture;
	int _width, _height;
	GLuint _glTexture;

	// Methods
public:
	PathTracer(const GLuint glTexture, const int pixelWidth, const int pixelHeight);
	void Update();
	void Draw();
	void Resize(const int pixelWidth, const int pixelHeight);
	~PathTracer();
private:
	inline static void _CheckCudaError(const cudaError_t cudaStatus, const char* functionName);
};

typedef unsigned int uint;

extern "C" __declspec(dllexport) IPathTracer* Create(const unsigned int glTexture, const int pixelWidth, const int pixelHeight) {
	gladLoadGL();
	return new PathTracer(glTexture, pixelWidth, pixelHeight);
}

extern "C" __declspec(dllexport) void Destroy(IPathTracer* pathTracer) {
	delete pathTracer;
}

