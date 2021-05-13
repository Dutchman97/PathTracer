#pragma once

#ifdef __NVCC__
#define GLM_FORCE_COMPILER_UNKNOWN
#endif

#include "glm/glm.hpp"
#include "glm/ext.hpp"

#ifdef __NVCC__
#undef GLM_FORCE_COMPILER_UNKNOWN
#endif

struct CameraData {
	glm::vec4 position = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
	glm::quat rotation = glm::quat(glm::vec3(0.0f, 0.0f, 0.0f));
	float fieldOfView = 90.0f; // Vertical field of view.

	inline bool operator==(const CameraData& other) {
		return position == other.position && rotation == other.rotation && fieldOfView == other.fieldOfView;
	}

	inline bool operator!=(const CameraData& other) {
		return !(*this == other);
	}
};
