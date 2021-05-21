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

__global__ void ResetCompactionArray(CompactionArray compactionArray) {
	compactionArray.Reset();
}

__global__ void InitializeRng(RngState* rngStates, int count) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= count) return;

	RNG_INIT(1337 + i, 0, i / 32, &rngStates[i]);
}

__global__ void InitializeRays(Ray* rays, RngState* rngStates, int screenWidth, int screenHeight, float4 origin, float4 topLeft, float4 bottomLeft, float4 bottomRight, Intersection* intersections, float4* stepBuffer, float4* frameBuffer, CompactionArray traverseSceneCompaction) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= screenWidth * screenHeight) return;

	uint x = i % screenWidth;
	uint y = i / screenWidth;

	float xScreen = ((float)x + RNG_GET_UNIFORM(&rngStates[i])) / screenWidth;
	float yScreen = ((float)y + RNG_GET_UNIFORM(&rngStates[i])) / screenHeight;

	Ray* rayPtr = &rays[i];
	rayPtr->origin = origin;
	rayPtr->direction = normalize(bottomLeft + (bottomRight - bottomLeft) * xScreen + (topLeft - bottomLeft) * yScreen);

	intersections[i] = NO_INTERSECTION;

	stepBuffer[i] = float4 { 1.0f, 1.0f, 1.0f, 1.0f };
	frameBuffer[i] = float4 { 0.0f, 0.0f, 0.0f, 1.0f };

	if (i == 0) {
		traverseSceneCompaction.data[0] = screenWidth * screenHeight;
	}
	traverseSceneCompaction.data[i + 1] = i;
}

__global__ void TraverseScene(Ray* rays, Triangle* triangles, int triangleCount, Vertex* vertices, Intersection* intersections, CompactionArray intersectCompaction, CompactionArray traverseSceneCompaction) {
	uint laneIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (laneIdx >= traverseSceneCompaction.GetCount()) return;
	uint rayIdx = traverseSceneCompaction.Get(laneIdx);

	for (int triangleIdx = 0; triangleIdx < triangleCount; triangleIdx++) {
		Intersection intersection = RayIntersectsTriangle(&rays[rayIdx], &triangles[triangleIdx], vertices);
		if (intersection.t > EPSILON && intersection.t < intersections[rayIdx].t) {
			intersections[rayIdx] = intersection;
			intersectCompaction.Add(rayIdx);
		}
	}
}

__global__ void Intersect(Ray* rays, Intersection* intersections, Material* materials, RngState* rngStates, float4* stepBuffer, float4* frameBuffer, CompactionArray intersectCompaction, CompactionArray traverseSceneCompaction) {
	uint laneIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (laneIdx >= intersectCompaction.GetCount()) return;
	uint rayIdx = intersectCompaction.Get(laneIdx);

	Material* materialPtr = &materials[intersections[rayIdx].materialIdx];
	float4 materialColor = materialPtr->color;

	if (materialPtr->type == Material::MaterialType::DIFFUSE) {
		float4 reflection = GetDiffuseReflection(intersections[rayIdx].normal, &rngStates[rayIdx]);
		rays[rayIdx].origin += rays[rayIdx].direction * intersections[rayIdx].t + reflection * EPSILON;
		rays[rayIdx].direction = reflection;

		stepBuffer[rayIdx] *= dot(intersections[rayIdx].normal, reflection) * 2.0f * materialColor;

		traverseSceneCompaction.Add(rayIdx);
	}
	else if (materialPtr->type == Material::MaterialType::EMISSIVE) {
		frameBuffer[rayIdx] = stepBuffer[rayIdx] * materialColor;
	}
	else {
		stepBuffer[rayIdx] *= make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	}
}
