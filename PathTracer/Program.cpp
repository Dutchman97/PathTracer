#include "Program.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <IPathTracerManagement.h>

GLFWwindow* Program::_glfwWindow;
int Program::_windowWidth, Program::_windowHeight;
GLuint Program::_shaderProgram;
IPathTracer* Program::_pathTracer;
Surface* Program::_mainSurface;
DestroyPathTracerFunc Program::_DestroyPathTracer;

constexpr GLuint VAO_COUNT = 128;

void Program::Initialize(const int windowWidth, const int windowHeight) {
	Program::_windowWidth = windowWidth;
	Program::_windowHeight = windowHeight;

	Program::_InitializeGlfw();
	Program::_InitializeGlad();

	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(Program::_MessageCallback, 0);

	Program::_shaderProgram = Program::_CreateShaderProgram("shaders/shader.vert", "shaders/shader.frag");

	Program::_mainSurface = new Surface(0.0f, 0.0f, 1.0f, 1.0f, windowWidth, windowHeight);
	//Program::_pathTracer = new PathTracer(Program::_mainSurface->GetTexture(), windowWidth, windowHeight);
}

void Program::Terminate() {
	//delete Program::_pathTracer;
	delete Program::_mainSurface;
	glfwTerminate();
}

bool Program::ShouldTerminate() {
	return glfwWindowShouldClose(Program::_glfwWindow);
}

void Program::Update() {
	glfwPollEvents();
	if (Program::_pathTracer != nullptr) {
		// Can't put this before polling events as that's when
		// the window resizing callback function gets called
		// and thus also recreating the OpenGL texture.
		Program::_pathTracer->BeginDrawing();
		Program::_pathTracer->Update();
	}
}

void Program::Draw() {
	// Set the color to clear with and clear the color buffer.
	glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// Draw all objects.
	glUseProgram(Program::_shaderProgram);
	if (Program::_pathTracer != nullptr) {
		Program::_pathTracer->FinalizeDrawing();
	}
	Program::_mainSurface->Draw();

	// Swap the back and front buffers.
	glfwSwapBuffers(Program::_glfwWindow);
}

#pragma region Private methods
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

	if (Program::_pathTracer != nullptr) {
		Program::_pathTracer->Resize(newWidth, newHeight);
	}
	Program::_mainSurface->Resize(newWidth, newHeight);
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
#define DLL_PROJECT "PT_CudaSimple"
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(Program::_glfwWindow, true);
	}
	else if (key == GLFW_KEY_0 && action == GLFW_PRESS) {
		CreatePathTracerFunc CreatePathTracer;
		bool loadLibrarySuccess = LoadPathTracerLibrary(DLL_PROJECT, &CreatePathTracer, &Program::_DestroyPathTracer);

		if (loadLibrarySuccess) {
			std::cout << "Loaded library " << DLL_PROJECT << std::endl;
			Program::_pathTracer = CreatePathTracer(Program::_mainSurface->GetTexture(), Program::_windowWidth, Program::_windowHeight);
		}
		else {
			std::cout << "Could not load library " << DLL_PROJECT << std::endl;
		}
	}
	else if (key == GLFW_KEY_9 && action == GLFW_PRESS) {
		if (Program::_pathTracer != nullptr) {
			Program::_DestroyPathTracer(Program::_pathTracer);
			Program::_pathTracer = nullptr;
		}
		bool unloadLibrarySuccess = UnloadPathTracerLibrary(DLL_PROJECT);

		if (unloadLibrarySuccess) {
			std::cout << "Unloaded library " << DLL_PROJECT << std::endl;
		}
		else {
			std::cout << "Could not unload library " << DLL_PROJECT << std::endl;
		}
	}
}

void GLAPIENTRY Program::_MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
	std::string sourceStr;
	switch (source) {
	case GL_DEBUG_SOURCE_API:
		sourceStr = "OpenGL API call";
		break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
		sourceStr = "window-system API";
		break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER:
		sourceStr = "shader compiler";
		break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:
		sourceStr = "third-party application";
		break;
	case GL_DEBUG_SOURCE_APPLICATION:
		sourceStr = "user-generated";
		break;
	case GL_DEBUG_SOURCE_OTHER:
		sourceStr = "unknown source";
		break;
	}

	std::string typeStr;
	switch (type) {
	case GL_DEBUG_TYPE_ERROR:
		typeStr = "error";
		break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
		typeStr = "deprecated behavior";
		break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
		typeStr = "undefined behavior";
		break;
	case GL_DEBUG_TYPE_PORTABILITY:
		typeStr = "not portable";
		break;
	case GL_DEBUG_TYPE_PERFORMANCE:
		typeStr = "performance issue";
		break;
	case GL_DEBUG_TYPE_MARKER:
		typeStr = "marker";
		break;
	case GL_DEBUG_TYPE_PUSH_GROUP:
		typeStr = "push to group";
		break;
	case GL_DEBUG_TYPE_POP_GROUP:
		typeStr = "pop from group";
		break;
	case GL_DEBUG_TYPE_OTHER:
		typeStr = "unknown type";
	}

	std::string severityStr;
	switch (severity) {
	case GL_DEBUG_SEVERITY_HIGH:
		severityStr = "HIGH";
		break;
	case GL_DEBUG_SEVERITY_MEDIUM:
		severityStr = "MEDIUM";
		break;
	case GL_DEBUG_SEVERITY_LOW:
		severityStr = "LOW";
		break;
	case GL_DEBUG_SEVERITY_NOTIFICATION:
		severityStr = "INFO";
		break;
	}

	fprintf(stderr, "GL CALLBACK (%s): source: '%s', type: '%s' - message:\n\t%s\n",
		severityStr.c_str(), sourceStr.c_str(), typeStr.c_str(), message);
}
#pragma endregion
