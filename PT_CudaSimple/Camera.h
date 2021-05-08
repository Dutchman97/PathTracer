#pragma once

#define GLM_FORCE_COMPILER_UNKNOWN
#include <glm/gtc/quaternion.hpp>
#include <glm/geometric.hpp>
#include <glm/vec4.hpp>
#include <glm/trigonometric.hpp>
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

	inline bool TryUpdate(const CameraData* data) {
		if (_data != *data) {
			_data = *data;
			return true;
		}
		return false;
	}

	inline void Resize(const int width, const int height) {
		_aspectRatio = (float)width / height;
	}

	inline float4 GetPosition() const {
		return *(float4*)&_data.position;
	}

	inline float4 GetRotation() const {
		glm::vec3 eulerAngles = glm::eulerAngles(_data.rotation);
		return make_float4(eulerAngles.x, eulerAngles.y, eulerAngles.z, 0.0f);
	}

	inline float4 GetTopLeft() const {
		glm::vec4 localTopLeft = _GetForwardVector() + 0.5f * GLM_UP_VECTOR - 0.5f * _aspectRatio * GLM_RIGHT_VECTOR;
		glm::vec4 direction = GLM_ROTATE_VECTOR(localTopLeft, _data.rotation);
		glm::vec4 glmResult = _data.position + direction;
		return *(float4*)&glmResult;
	}

	inline float4 GetBottomLeft() const {
		glm::vec4 localBottomLeft = _GetForwardVector() - 0.5f * GLM_UP_VECTOR - 0.5f * _aspectRatio * GLM_RIGHT_VECTOR;
		glm::vec4 direction = GLM_ROTATE_VECTOR(localBottomLeft, _data.rotation);
		glm::vec4 glmResult = _data.position + direction;
		return *(float4*)&glmResult;
	}

	inline float4 GetBottomRight() const {
		glm::vec4 localBottomRight = _GetForwardVector() - 0.5f * GLM_UP_VECTOR + 0.5f * _aspectRatio * GLM_RIGHT_VECTOR;
		glm::vec4 direction = GLM_ROTATE_VECTOR(localBottomRight, _data.rotation);
		glm::vec4 glmResult = _data.position + direction;
		return *(float4*)&glmResult;
	}

private:
	inline glm::vec4 _GetForwardVector() const {
		return glm::vec4(0.0f, 0.0f, 2.0f / glm::tan(glm::radians(_data.fieldOfView * 0.5f)), 0.0f);
	}
};

