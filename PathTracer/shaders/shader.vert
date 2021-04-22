#version 440 core

layout (location = 0) in vec2 aPosition;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aUv;

out vec3 color;
out vec2 uv;

void main() {
	gl_Position = vec4(aPosition, 0.0f, 1.0f);
	color = aColor;
	uv = aUv;
}
