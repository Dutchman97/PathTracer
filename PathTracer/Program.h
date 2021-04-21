#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

class Program {
	// Properties
private:
	static GLFWwindow* _glfwWindow;
	static int _windowWidth, _windowHeight;

	static GLuint* _vertexArrayObjects;
	static GLuint _shaderProgram;
	static GLuint _texture;


	// Methods
public:
	static void Initialize(const int windowWidth, const int windowHeight);
	static void Terminate();
	static bool ShouldTerminate();
	static void Update();
	static void Draw();

private:
	static void _InitializeGlfw();
	static void _InitializeGlad();
	static GLuint _CompileShader(const char* filePath, const GLenum glShaderType);
	static GLuint _CreateShaderProgram(const char* vertexShaderPath, const char* fragmentShaderPath);
	static void _CreateVertexArrayObjects(GLsizei count, GLuint* vertexArrayObjects);
	
	static void _FramebufferResizeCallback(GLFWwindow* window, int newWidth, int newHeight);
	static void _KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mod);
};

