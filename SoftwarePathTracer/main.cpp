#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

int windowWidth  = 1920 / 2;
int windowHeight = 1080 / 2;

/// <summary>
/// Perform any logic required before terminating the program.
/// </summary>
void terminate() {
	glfwTerminate();
}

/// <summary>
/// Callback function for when the framebuffer gets resized.
/// </summary>
/// <param name="window">GLFW window object.</param>
/// <param name="newWidth">New width of the window.</param>
/// <param name="newHeight">New height of the window.</param>
void framebufferResizeCallback(GLFWwindow* window, int newWidth, int newHeight) {
	windowWidth  = newWidth;
	windowHeight = newHeight;
	glViewport(0, 0, newWidth, newHeight);
}

int main() {
	// Initialize the GLFW window
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Path Tracer", nullptr, nullptr);
	if (window == nullptr) {
		std::cout << "Unable to create GLFW window." << std::endl;
		terminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize glad and OpenGL
	int loadGlSuccess = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	if (!loadGlSuccess) {
		std::cout << "Unable to initialize glad." << std::endl;
		terminate();
		return -1;
	}
	glViewport(0, 0, windowWidth, windowHeight);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	

	// Main loop
	while (!glfwWindowShouldClose(window)) {
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	terminate();
	return 0;
}
