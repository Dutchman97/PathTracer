#pragma once

#include "CameraData.h"

class IPathTracer {
public:
	virtual void Update(const CameraData* cameraData) = 0;
	virtual void BeginDrawing() = 0;
	virtual void FinalizeDrawing() = 0;
	virtual void Resize(const int pixelWidth, const int pixelHeight) = 0;
};

typedef IPathTracer* (*CreatePathTracerFunc)(const unsigned int glTexture, const int pixelWidth, const int pixelHeight, const CameraData* cameraData);
typedef void (*DestroyPathTracerFunc)(IPathTracer*);
