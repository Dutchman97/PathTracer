#include "Program.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

GLFWwindow* Program::_glfwWindow;
int Program::_windowWidth, Program::_windowHeight;
GLuint* Program::_vertexArrayObjects;
GLuint Program::_shaderProgram;
GLuint Program::_texture;

constexpr GLuint VAO_COUNT = 5;

void Program::Initialize(const int windowWidth, const int windowHeight) {
	Program::_windowWidth = windowWidth;
	Program::_windowHeight = windowHeight;

	Program::_InitializeGlfw();
	Program::_InitializeGlad();

	Program::_shaderProgram = Program::_CreateShaderProgram("shaders/shader.vert", "shaders/shader.frag");

	Program::_vertexArrayObjects = (GLuint*)malloc(sizeof(GLuint) * VAO_COUNT);
	Program::_CreateVertexArrayObjects(VAO_COUNT, Program::_vertexArrayObjects);


	//GLuint texture;
	glGenTextures(1, &Program::_texture);
	glBindTexture(GL_TEXTURE_2D, Program::_texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	GLubyte* textureData = (GLubyte*)malloc(windowWidth * windowHeight * 3 * sizeof(GLubyte));
	for (int y = 0; y < windowHeight; y++) {
		for (int x = 0; x < windowWidth; x++) {
			textureData[(x + y * windowWidth) * 3 + 0] = 255;
			textureData[(x + y * windowWidth) * 3 + 1] = 127;
			textureData[(x + y * windowWidth) * 3 + 2] = 127;
		}
	}
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, NULL, GL_RGB, GL_UNSIGNED_BYTE, textureData);
	glBindTexture(GL_TEXTURE_2D, NULL);
	free(textureData);
}

void Program::Terminate() {
	glfwTerminate();
	free(Program::_vertexArrayObjects);
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

	int second = (int)glfwGetTime();
	int vaoIdx = second % VAO_COUNT;

	glUseProgram(Program::_shaderProgram);
	glBindTexture(GL_TEXTURE_2D, Program::_texture);
	glBindVertexArray(Program::_vertexArrayObjects[vaoIdx]);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

	glfwSwapBuffers(Program::_glfwWindow);
}

#pragma region Private methods
void Program::_CreateVertexArrayObjects(GLsizei count, GLuint* vertexArrayObjects) {
	glGenVertexArrays(count, vertexArrayObjects);

	// This works but can be further optimized.
	for (GLsizei i = 0; i < count; i++) {

		float col = (i + 1) / (float)count;
		GLfloat vertices[] = {
			// Position        |  Color          |  UV
			-0.9f, -0.9f, 0.0f,  col, 0.0f, 0.0f, 0.0f, 0.0f, // Bottom-left
			-0.9f,  0.9f, 0.0f, 0.0f,  col, 0.0f, 1.0f, 0.0f, // Bottom-right
			 0.9f,  0.9f, 0.0f, 0.0f, 0.0f,  col, 1.0f, 1.0f, // Top-right
			 0.9f, -0.9f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, // Top-left
		};
		GLuint indices[] = {
			0, 1, 2,
			2, 3, 0,
		};

		glBindVertexArray(vertexArrayObjects[i]);

		GLuint vertexBufferObject, elementBufferObject;
		glGenBuffers(1, &vertexBufferObject);
		glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObject);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glGenBuffers(1, &elementBufferObject);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, elementBufferObject);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);

		glBindVertexArray(NULL);
		glBindBuffer(GL_ARRAY_BUFFER, NULL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, NULL);
	}
}

GLuint Program::_CreateShaderProgram(const char* vertexShaderPath, const char* fragmentShaderPath) {
	GLuint vertexShader   = Program::_CompileShader(vertexShaderPath,   GL_VERTEX_SHADER  );
	GLuint fragmentShader = Program::_CompileShader(fragmentShaderPath, GL_FRAGMENT_SHADER);

	// Create a shader program and bundle the shaders in this program.
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// Check if the program was linked successfully.
	GLint success;
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		GLchar infoLog[512];
		GLsizei infoLength;
		glGetProgramInfoLog(shaderProgram, 512 * sizeof(GLchar), &infoLength, infoLog);
		std::cout << "Failed to link shader program";
		if (infoLength > 512 * sizeof(GLchar)) {
			std::cout << " (512/" << infoLength / sizeof(GLchar) << ")";
		}
		std::cout << std::endl << infoLog << std::endl;
		throw std::exception("Failed to link shader program");
	}

	// Shaders are now linked to the program and will not be used individually anymore.
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}

GLuint Program::_CompileShader(const char* filePath, GLenum glShaderType) {
	// Read the file.
	std::ifstream shaderFile;
	try {
		shaderFile.open(filePath);
	}
	catch (std::exception e) {
		std::cerr << "Unable to open shader file \"" << filePath << "\"" << std::endl << e.what() << std::endl;
		throw e;
	}

	// Place the file's contents into a C string.
	std::stringstream shaderStream;
	shaderStream << shaderFile.rdbuf();
	const std::string shaderCode = shaderStream.str();
	const char* shaderCodeC = shaderCode.c_str();

	// Compile the shader.
	GLuint shader = glCreateShader(glShaderType);
	glShaderSource(shader, 1, &shaderCodeC, nullptr);
	glCompileShader(shader);

	// Check if the shader was compiled successfully.
	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		GLchar infoLog[512];
		GLsizei infoLength;
		glGetShaderInfoLog(shader, 512 * sizeof(GLchar), &infoLength, infoLog);
		std::cout << "Failed to compile shader \"" << filePath << "\"";
		if (infoLength > 512 * sizeof(GLchar)) {
			std::cout << " (512/" << infoLength / sizeof(GLchar) << ")";
		}
		std::cout << std::endl << infoLog << std::endl;
		throw std::exception("Failed to compile shader");
	}
	return shader;
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
#pragma endregion
