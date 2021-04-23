#pragma once

#include <glad/glad.h>

class PathTracer {
	// Properties
private:
	GLuint _texture;


	// Methods
public:
	PathTracer(GLuint texture, const int pixelWidth, const int pixelHeight);
	void Update();
	void Draw();
	void Resize(const int pixelWidth, const int pixelHeight);
};

