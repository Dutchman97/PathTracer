#pragma once

#include <IPathTracer.h>
#include <glad/glad.h>

class PathTracer : public IPathTracer {
public:
	PathTracer();
	void Update();
	void Draw();
	void Resize(const int pixelWidth, const int pixelHeight);
	~PathTracer();
};

extern "C" __declspec(dllexport) IPathTracer* Create(const unsigned int glTexture, const int pixelWidth, const int pixelHeight) {
	gladLoadGL();
	return new PathTracer();
}

extern "C" __declspec(dllexport) void Destroy(IPathTracer* pathTracer) {
	delete pathTracer;
}

