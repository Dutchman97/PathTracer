#version 440 core

in vec3 color;
in vec2 uv;

out vec4 FragColor;

uniform sampler2D diffuseTexture;

void main() {
	FragColor = vec4(color, 1.0f) * texture(diffuseTexture, uv);
}
