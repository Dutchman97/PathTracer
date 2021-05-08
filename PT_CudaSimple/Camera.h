#pragma once

#define GLM_FORCE_COMPILER_UNKNOWN
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/quaternion_common.hpp>
#include <glm/ext/quaternion_transform.hpp>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/ext/quaternion_relational.hpp>
#include <glm/ext/quaternion_exponential.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#include <CameraData.h>
#include <cuda_runtime.h>
#undef GLM_FORCE_COMPILER_UNKNOWN

#define GLM_RIGHT_VECTOR glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)
#define GLM_UP_VECTOR glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)
#define GLM_FORWARD_VECTOR glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)

#define GLM_ROTATE_VECTOR(vector, rotation) (rotation * vector * glm::conjugate(rotation))

class Camera {
	// Properties
private:
	CameraData _data;
	float _aspectRatio; // Height divided by width.

	// Methods
public:
	Camera(const CameraData* cameraData, const float aspectRatio) : _data(*cameraData), _aspectRatio(aspectRatio) {

	}

	bool TryUpdate(const CameraData* data) {
		if (_data != *data) {
			_data = *data;
			return true;
		}
		return false;
	}

	inline void Resize(const int width, const int height) {
		_aspectRatio = (float)width / height;
	}

	inline float4 Position() const {
		return *(float4*)&_data.position;
	}

	inline float4 TopLeft() const {
		glm::vec4 localTopLeft = _ForwardVector() + 0.5f * GLM_UP_VECTOR - 0.5f * _aspectRatio * GLM_RIGHT_VECTOR;
		glm::vec4 lookDirection = GLM_ROTATE_VECTOR(localTopLeft, _data.rotation);
		glm::vec4 glmResult = _data.position + lookDirection;
		return make_float4(glmResult.x, glmResult.y, glmResult.z, glmResult.w);
	}

	inline float4 TopRight() const {
		glm::vec4 localTopRight = _ForwardVector() + 0.5f * GLM_UP_VECTOR + 0.5f * _aspectRatio * GLM_RIGHT_VECTOR;
		glm::vec4 lookDirection = GLM_ROTATE_VECTOR(localTopRight, _data.rotation);
		glm::vec4 glmResult = _data.position + lookDirection;
		return make_float4(glmResult.x, glmResult.y, glmResult.z, glmResult.w);
	}

	inline float4 BottomLeft() const {
		glm::vec4 localTopRight = _ForwardVector() - 0.5f * GLM_UP_VECTOR - 0.5f * _aspectRatio * GLM_RIGHT_VECTOR;
		glm::vec4 lookDirection = GLM_ROTATE_VECTOR(localTopRight, _data.rotation);
		glm::vec4 glmResult = _data.position + lookDirection;
		return make_float4(glmResult.x, glmResult.y, glmResult.z, glmResult.w);
	}

private:
	inline glm::vec4 _ForwardVector() const {
		return glm::vec4(0.0f, 0.0f, 2.0f / glm::tan(glm::radians(_data.fieldOfView * 0.5f)), 0.0f);
	}
};

