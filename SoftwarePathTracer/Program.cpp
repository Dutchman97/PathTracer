#include "Program.h"

#include <iostream>
#include <stdexcept>

GLFWwindow* Program::_glfwWindow;
int Program::_windowWidth, Program::_windowHeight;

void Program::Initialize(const int windowWidth, const int windowHeight) {
	Program::_windowWidth = windowWidth;
	Program::_windowHeight = windowHeight;

	Program::_InitializeGlfw();
	Program::_InitializeGlad();
}

void Program::Terminate() {
	glfwTerminate();
}

bool Program::ShouldTerminate() {
	return glfwWindowShouldClose(Program::_glfwWindow);
}

void Program::Update() {
	glfwPollEvents();
}

void Program::Draw() {
	// Set the color to clear with and clear the color buffer.
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glfwSwapBuffers(Program::_glfwWindow);
}

void Program::_InitializeGlfw() {
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(Program::_windowWidth, Program::_windowHeight, "Path Tracer", nullptr, nullptr);
	Program::_glfwWindow = window;
	if (window == nullptr) {
		Program::Terminate();
		throw std::exception("Unable to create GLFW window.");
	}
	glfwMakeContextCurrent(window);
}

void Program::_InitializeGlad() {
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		Program::Terminate();
		throw std::exception("Unable to initialize OpenGL with glad.");
	}
	glViewport(0, 0, Program::_windowWidth, Program::_windowHeight);
	glfwSetFramebufferSizeCallback(Program::_glfwWindow, Program::_FramebufferResizeCallback);
	glfwSetKeyCallback(Program::_glfwWindow, Program::_KeyCallback);
}

/// <summary>
/// Callback function for when the framebuffer gets resized.
/// </summary>
/// <param name="window">GLFW window object.</param>
/// <param name="newWidth">New width of the window.</param>
/// <param name="newHeight">New height of the window.</param>
void Program::_FramebufferResizeCallback(GLFWwindow* window, int newWidth, int newHeight) {
	Program::_windowWidth = newWidth;
	Program::_windowHeight = newHeight;
	glViewport(0, 0, newWidth, newHeight);
}

/// <summary>
/// GLFW callback function for key input.
/// </summary>
/// <param name="window">GLFW window object.</param>
/// <param name="key">The keyboard key that was pressed or released.</param>
/// <param name="scancode">The system-specific scancode of the key.</param>
/// <param name="action">GLFW_PRESS, GLFW_RELEASE, or GLFW_REPEAT.</param>
/// <param name="mod">Bit field describing which modifier keys were held down.</param>
void Program::_KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mod) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(Program::_glfwWindow, true);
	}
}
