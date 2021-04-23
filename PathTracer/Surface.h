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
	void Resize(const int windowPixelWidth, const int windowPixelHeight);
	void Release();
	~Surface();

	// Forbid copying a Surface object, and grab resources from the moved-from object when moving a Surface.
	// https://www.khronos.org/opengl/wiki/Common_Mistakes#RAII_and_hidden_destructor_calls
	Surface(const Surface&) = delete;
	Surface& operator=(const Surface&) = delete;
	Surface(Surface&& other) noexcept {
		this->_texture = other._texture;
		other._texture = NULL;
		this->_vertexArrayObject = other._vertexArrayObject;
		other._vertexArrayObject = NULL;

		this->_width = other._width;
		this->_height = other._height;
	}
	Surface& operator=(Surface&& other) noexcept {
		if (this != &other) {
			this->Release();
			std::swap(this->_texture, other._texture);
			std::swap(this->_vertexArrayObject, other._vertexArrayObject);
			this->_width = other._width;
			this->_height = other._height;
		}
	}

	/// <summary>
	/// Gets the OpenGL texture ID of the surface.
	/// </summary>
	/// <returns></returns>
	inline GLuint GetTexture() { return this->_texture; }

private:
	GLuint _CreateVertexArrayObject(const float x, const float y, const float width, const float height);
	GLuint _CreateTexture(const int pixelWidth, const int pixelHeight);
	void _SetTextureData(const GLuint texture, const int pixelWidth, const int pixelHeight);
};

