#pragma once

#include <glad/glad.h>
#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

class Surface {
	// Properties
public:
	struct Vertex {
		glm::vec2 position;
		glm::vec3 color;
		glm::vec2 uv;
	};

private:
	GLuint _vertexArrayObject;
	GLuint _texture;

	float _width, _height;


	// Methods
public:
	Surface(const float x, const float y, const float width, const float height, const int windowPixelWidth, const int windowPixelHeight);
	void Draw();
	~Surface();

	/// <summary>
	/// Get the OpenGL texture ID of the surface.
	/// </summary>
	/// <returns></returns>
	inline GLuint GetTexture() { return this->_texture; }

private:
	GLuint _CreateVertexArrayObject(const float x, const float y, const float width, const float height);
	GLuint _CreateTexture(const int pixelWidth, const int pixelHeight);
};

