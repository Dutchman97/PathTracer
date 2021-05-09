#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <IPathTracer.h>
#include <CameraData.h>
#include "Surface.h"

using namespace std::chrono;

class Program {
	// Properties
private:
	static GLFWwindow* _glfwWindow;
	static int _windowWidth, _windowHeight;
	
	static IPathTracer* _pathTracer;
	static DestroyPathTracerFunc _DestroyPathTracer;
	static Surface* _mainSurface;
	static CameraData* _cameraData;

	static GLuint _shaderProgram;

	static time_point<steady_clock> _lastFrame;



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
	
	static void _FramebufferResizeCallback(GLFWwindow* window, int newWidth, int newHeight);
	static void _KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mod);
	static void GLAPIENTRY _MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
};

