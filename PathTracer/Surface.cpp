#include "Surface.h"

#include <stdexcept>
#include <math.h>
#include <iostream>

/// <summary>
/// An OpenGL wrapper class representing a quad with a texture.
/// </summary>
/// <param name="x">The x-coordinate of the bottom-left corner of the surface, proportional to the window.</param>
/// <param name="y">The y-coordinate of the bottom-left corner of the surface, proportional to the window.</param>
/// <param name="width">The width of the surface, proportional to the window.</param>
/// <param name="height">The height of the surface, proportional to the window.</param>
/// <param name="windowPixelWidth">The width of the window in pixels.</param>
/// <param name="windowPixelHeight">The height of the window in pixels.</param>
Surface::Surface(const float x, const float y, const float width, const float height, const int windowPixelWidth, const int windowPixelHeight) {
	this->_width = width;
	this->_height = height;

	this->_vertexArrayObject = this->_CreateVertexArrayObject(x, y, width, height);
	this->_texture = this->_CreateTexture((int)roundf(width * windowPixelWidth), (int)roundf(height * windowPixelHeight));
}

/// <summary>
/// Destroys this object and releases all OpenGL resources managed by this object.
/// </summary>
Surface::~Surface() {
	this->Release();
}

/// <summary>
/// Destroys all OpenGL resources managed by this object.
/// </summary>
void Surface::Release() {
	glDeleteTextures(1, &this->_texture);
	glDeleteVertexArrays(1, &this->_vertexArrayObject);
}

/// <summary>
/// Resizes the surface's texture so that it fits exactly in the same region of the resized window again.
/// The size of the surface itself is proportional to the window, and as such does not need resizing.
/// Call this method whenever the window gets resized.
/// </summary>
/// <param name="windowPixelWidth">The new width of the window.</param>
/// <param name="windowPixelHeight">The new height of the window.</param>
void Surface::Resize(const int windowPixelWidth, const int windowPixelHeight) {
	this->_SetTextureData(this->_texture, (int)roundf(this->_width * windowPixelWidth), (int)roundf(this->_height * windowPixelHeight));
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

	this->_SetTextureData(texture, pixelWidth, pixelHeight);

	// Clean up.
	glBindTexture(GL_TEXTURE_2D, NULL);

	return texture;
}

/// <summary>
/// Sets the texture's data to be fully white and in the given dimensions.
/// </summary>
/// <param name="texture">The OpenGL texture ID of the texture.</param>
/// <param name="pixelWidth">The width of the texture.</param>
/// <param name="pixelHeight">The height of the texture.</param>
void Surface::_SetTextureData(const GLuint texture, const int pixelWidth, const int pixelHeight) {
	// Assign memory for the color data of the texture.
	GLubyte* textureData = (GLubyte*)_aligned_malloc(pixelWidth * pixelHeight * 4 * sizeof(GLubyte), 4);
	if (!textureData) {
		throw std::exception("Unable to assign memory for texture data.");
	}

	// Set all pixels to white (all RGBA values to 255).
	for (int i = 0; i < pixelWidth * pixelHeight * 4; i++) {
		textureData[i] = 255;
	}

	// Set the texture's data.
	// https://www.khronos.org/opengl/wiki/Common_Mistakes#Texture_upload_and_pixel_reads
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, pixelWidth, pixelHeight, NULL, GL_RGBA, GL_UNSIGNED_BYTE, textureData);
	_aligned_free(textureData);
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
