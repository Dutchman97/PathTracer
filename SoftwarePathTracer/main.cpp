#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

int windowWidth  = 1920 / 2;
int windowHeight = 1080 / 2;

/// <summary>
/// Perform any logic required before terminating the program.
/// </summary>
inline void terminateProgram() {
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

/// <summary>
/// GLFW callback function for key input.
/// </summary>
/// <param name="window">GLFW window object.</param>
/// <param name="key">The keyboard key that was pressed or released.</param>
/// <param name="scancode">The system-specific scancode of the key.</param>
/// <param name="action">GLFW_PRESS, GLFW_RELEASE, or GLFW_REPEAT.</param>
/// <param name="mod">Bit field describing which modifier keys were held down.</param>
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mod) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, true);
	}
}

int main() {
	// Initialize the GLFW window.
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Path Tracer", nullptr, nullptr);
	if (window == nullptr) {
		std::cout << "Unable to create GLFW window." << std::endl;
		terminateProgram();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Initialize glad and OpenGL.
	int loadGlSuccess = gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
	if (!loadGlSuccess) {
		std::cout << "Unable to initialize glad." << std::endl;
		terminateProgram();
		return -1;
	}
	glViewport(0, 0, windowWidth, windowHeight);
	glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
	glfwSetKeyCallback(window, keyCallback);
	

	// Main loop.
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();

		// Set the color to clear with and clear the color buffer.
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glfwSwapBuffers(window);
	}

	terminateProgram();
	return 0;
}
