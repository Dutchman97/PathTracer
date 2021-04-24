#pragma once

class IPathTracer {
public:
	virtual void Update() = 0;
	virtual void Draw() = 0;
	virtual void Resize(const int pixelWidth, const int pixelHeight) = 0;
};
