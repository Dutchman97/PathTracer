#pragma once

class IPathTracer {
public:
	virtual void Update() = 0;
	virtual void Draw() = 0;
	virtual void Resize(const int pixelWidth, const int pixelHeight) = 0;
};

typedef IPathTracer* (*CreatePathTracerFunc)(const unsigned int glTexture, const int pixelWidth, const int pixelHeight);
typedef void (*DestroyPathTracerFunc)(IPathTracer*);
