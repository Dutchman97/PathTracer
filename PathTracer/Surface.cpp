#include "Surface.h"

#include <stdexcept>
#include <math.h>
#include <iostream>

Surface::Surface(const float x, const float y, const float width, const float height, const int windowPixelWidth, const int windowPixelHeight) {
	this->_width = width;
	this->_height = height;

	this->_vertexArrayObject = this->_CreateVertexArrayObject(x, y, width, height);
	this->_texture = this->_CreateTexture((int)roundf(width * windowPixelWidth), (int)roundf(height * windowPixelHeight));
}

Surface::~Surface() {
	this->Release();
}

void Surface::Release() {
	glDeleteTextures(1, &this->_texture);
	glDeleteVertexArrays(1, &this->_vertexArrayObject);
}

/// <summary>
/// Draw the surface.
/// </summary>
void Surface::Draw() {
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, this->_texture);
	glBindVertexArray(this->_vertexArrayObject);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

#pragma region Private methods
/// <summary>
/// Creates an OpenGL RGB16F texture of the specified size.
/// </summary>
/// <param name="pixelWidth">The width of the texture in pixels.</param>
/// <param name="pixelHeight">The height of the texture in pixels.</param>
/// <returns>The OpenGL texture ID.</returns>
GLuint Surface::_CreateTexture(const int pixelWidth, const int pixelHeight) {
	// Create a texture and use it as texture0.
	GLuint texture;
	glGenTextures(1, &texture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture);

	// Set the texture's parameters.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	// Assign memory for the color data of the texture.
	GLubyte* textureData = (GLubyte*)_aligned_malloc(pixelWidth * pixelHeight * 4 * sizeof(GLubyte), 4);
	if (!textureData) {
		throw std::exception("Unable to assign memory for texture data.");
	}

	// Set all pixels to white.
#pragma warning (disable : 6386)
	for (int i = 0; i < pixelWidth * pixelHeight; i++) {
		textureData[i * 4 + 0] = 255;
		textureData[i * 4 + 1] = 255; // This line gives a C6386 warning for some reason?
		textureData[i * 4 + 2] = 255;
		textureData[i * 4 + 3] = 255;
	}
#pragma warning (default : 6386)

	// Set the texture's data.
	// https://www.khronos.org/opengl/wiki/Common_Mistakes#Texture_upload_and_pixel_reads
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, pixelWidth, pixelHeight, NULL, GL_RGBA, GL_UNSIGNED_BYTE, textureData);

	// Clean up.
	glBindTexture(GL_TEXTURE_2D, NULL);
	_aligned_free(textureData);

	return texture;
}

/// <summary>
/// Creates a surface as an OpenGL vertex array object.
/// </summary>
/// <param name="x">Left border of the surface on the window [0, 1].</param>
/// <param name="y">Bottom border of the surface on the window [0, 1].</param>
/// <param name="width">Width of the surface as proportion of the window [0, 1].</param>
/// <param name="height">Height of the surface as proportion of the window [0, 1].</param>
/// <returns>The ID of the vertex array object.</returns>
GLuint Surface::_CreateVertexArrayObject(const float x, const float y, const float width, const float height) {
	GLuint vertexArrayObject;
	glGenVertexArrays(1, &vertexArrayObject);
	glBindVertexArray(vertexArrayObject);

	constexpr glm::vec2 uvBottomLeft(0.0f, 0.0f);
	constexpr glm::vec2 uvBottomRight(1.0f, 0.0f);
	constexpr glm::vec2 uvTopRight(1.0f, 1.0f);
	constexpr glm::vec2 uvTopLeft(0.0f, 1.0f);

	glm::vec2 bottomLeft (x        , y);
	glm::vec2 bottomRight(x + width, y);
	glm::vec2 topRight   (x + width, y + height);
	glm::vec2 topLeft    (x        , y + height);

	// Vertices of the surface (position, color, UV).
	Surface::Vertex vertices[] = {
		Vertex { bottomLeft  * 2.0f - 1.0f, glm::vec3(1.0f, 0.0f, 0.0f), uvBottomLeft  },
		Vertex { bottomRight * 2.0f - 1.0f, glm::vec3(0.0f, 1.0f, 0.0f), uvBottomRight },
		Vertex { topRight    * 2.0f - 1.0f, glm::vec3(0.0f, 0.0f, 1.0f), uvTopRight    },
		Vertex { topLeft     * 2.0f - 1.0f, glm::vec3(1.0f, 1.0f, 1.0f), uvTopLeft     },
	};
	GLuint indices[] = {
		0, 1, 2,
		2, 3, 0,
	};

	// Create the vertex buffer and set its data.
	GLuint vertexBufferObject, elementBufferObject;
	glGenBuffers(1, &vertexBufferObject);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	// Create the element buffer and set its data.
	glGenBuffers(1, &elementBufferObject);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBufferObject);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Set where each attribute is located in the data.
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (void*)0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 7 * sizeof(GLfloat), (void*)(5 * sizeof(GLfloat)));
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);

	// Clean-up.
	glBindVertexArray(NULL);
	glBindBuffer(GL_ARRAY_BUFFER, NULL);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, NULL);
	GLuint buffers[2] { vertexBufferObject, elementBufferObject };
	glDeleteBuffers(2, buffers);

	return vertexArrayObject;
}
#pragma endregion
