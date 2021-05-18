#include "kernels.cuh"
#include "Functions.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture, float4* frameBuffer, int screenWidth, int screenHeight, uint frameNumber) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= screenWidth * screenHeight) return;

	uint x = i % screenWidth;
	uint y = i / screenWidth;

	// IMPORTANT: Surface functions use bytes for addressing memory; x-coordinate is in bytes.
	// Y-coordinate does not need to be multiplied as the byte offset of the corresponding y-coordinate is internally calculated.
	float4 color = frameBuffer[i];
	float4 previousColor;
	surf2Dread(&previousColor, texture, x * sizeof(float4), y);

	float reciprocal = 1.0f / (frameNumber + 1);
	surf2Dwrite(color * reciprocal + previousColor * frameNumber * reciprocal, texture, x * sizeof(float4), y);
}

__global__ void InitializeRng(curandStateXORWOW_t* rngStates, int count) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= count) return;

	curand_init(1337 + i, 0, 0, &rngStates[i]);
}

__global__ void InitializeRays(Ray* rays, curandStateXORWOW_t* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 bottomLeft, float4 bottomRight, Intersection* intersections, float4* frameBuffer) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= screenWidth * screenHeight) return;

	uint x = i % screenWidth;
	uint y = i / screenWidth;

	float xScreen = ((float)x + curand_uniform(&rngStates[i])) / screenWidth;
	float yScreen = ((float)y + curand_uniform(&rngStates[i])) / screenHeight;

	Ray* rayPtr = &rays[i];
	rayPtr->origin = origin;
	rayPtr->direction = normalize(bottomLeft + (bottomRight - bottomLeft) * xScreen + (topLeft - bottomLeft) * yScreen);

	intersections[i] = NO_INTERSECTION;

	frameBuffer[i] = make_float4(1.0f, 1.0f, 1.0f, 1.0f);

	// if (i == 0) {
	//     traverseSceneCompaction.count = screenWidth * screenHeight;
	// }
	// traverseSceneCompaction.array[i] = i;
}

// Before TraverseScene:
// intersectionCompaction.count = 0;

__global__ void TraverseScene(Ray* rays, int rayCount, Triangle* triangles, int triangleCount, Vertex* vertices, Intersection* intersections) {
	uint rayIdx = threadIdx.x + blockIdx.x * blockDim.x; // laneIdx
	if (rayIdx >= rayCount || rays[rayIdx].direction == ZERO_VECTOR) return; // if (laneIdx >= traverseSceneCompaction.count) return;

	// uint rayIdx = traverseSceneCompaction.array[laneIdx];

	for (int triangleIdx = 0; triangleIdx < triangleCount; triangleIdx++) {
		Intersection intersection = RayIntersectsTriangle(&rays[rayIdx], &triangles[triangleIdx], vertices);
		if (intersection.t > EPSILON && intersection.t < intersections[rayIdx].t) {
			intersections[rayIdx] = intersection;

			// uint intersectionIdx = atomic_add(intersectionCompaction.count);
			// intersectionCompaction.array[intersectionIdx] = rayIdx;
		}
	}
}

// Before Intersect:
// traverseSceneCompaction.count = 0;

__global__ void Intersect(Ray* rays, int rayCount, Intersection* intersections, Material* materials, curandStateXORWOW_t* rngStates, float4* frameBuffer) {
	uint rayIdx = threadIdx.x + blockIdx.x * blockDim.x; // laneIdx
	if (rayIdx >= rayCount || rays[rayIdx].direction == ZERO_VECTOR) return; // if (laneIdx >= intersectCompaction.count) return;

	if (!intersections[rayIdx].Hit()) { // Remove when compaction implemented
		frameBuffer[rayIdx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		return;
	}

	// uint rayIdx = intersectCompaction.array[laneIdx]

	Material* materialPtr = &materials[intersections[rayIdx].materialIdx];
	float4 materialColor = materialPtr->color;
	float4 radiance;

	if (materialPtr->type == Material::MaterialType::DIFFUSE) {
		float4 reflection = GetDiffuseReflection(intersections[rayIdx].normal, &rngStates[rayIdx]);
		rays[rayIdx].origin += rays[rayIdx].direction * intersections[rayIdx].t + reflection * EPSILON;
		rays[rayIdx].direction = reflection;

		radiance = dot(intersections[rayIdx].normal, reflection) * 2.0f * materialColor;

		// uint traverseSceneIdx = atomic_add(traverseSceneCompaction.count);
		// traverseSceneCompaction.array[traverseSceneIdx] = rayIdx;
	}
	else if (materialPtr->type == Material::MaterialType::EMISSIVE) {
		// Use this as "path finished" until compaction is implemented.
		rays[rayIdx].direction = ZERO_VECTOR;
		radiance = materialColor;
	}
	else {
		// Use this as "path finished" until compaction is implemented.
		rays[rayIdx].direction = ZERO_VECTOR;
		radiance = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	}

	frameBuffer[rayIdx] *= radiance;
}
