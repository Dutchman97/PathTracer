#include "kernels.cuh"

__global__ void DrawToTexture(cudaSurfaceObject_t texture, int screenWidth, int screenHeight, Intersection* intersections, uint frameNumber, float4* frameBuffer) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= screenWidth * screenHeight) return;

	uint x = i % screenWidth;
	uint y = i / screenWidth;

	//float t = intersections[i].t;
	//float red = (t > EPSILON && t < FLT_MAX) ? 1.0f : 0.2f;
	//float4 color = make_float4(red, 0.2f, 0.2f, 1.0f);

	float4 color = frameBuffer[i];

	// IMPORTANT: Surface functions use bytes for addressing memory; x-coordinate is in bytes.
	// Y-coordinate does not need to be multiplied as the byte offset of the corresponding y-coordinate is internally calculated.
	float4 previousColor;
	surf2Dread(&previousColor, texture, x * sizeof(float4), y);
	surf2Dwrite(color / (frameNumber + 1) + previousColor * frameNumber / (frameNumber + 1), texture, x * sizeof(float4), y);
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
}

// Uses the intersection algorithm by Möller and Trumbore.
// https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
__device__ Intersection RayIntersectsTriangle(Ray* rayPtr, Triangle* trianglePtr, Vertex* vertices) {
	int idx0 = trianglePtr->vertexIdx0;
	int idx1 = trianglePtr->vertexIdx1;
	int idx2 = trianglePtr->vertexIdx2;

	float4 v0 = vertices[idx0].position;
	float4 v1 = vertices[idx1].position;
	float4 v2 = vertices[idx2].position;

	float4 edge0 = v1 - v0;
	float4 edge1 = v2 - v0;

	float4 h = cross(rayPtr->direction, edge1);
	float determinant = dot(edge0, h);

#ifdef CULLING_ENABLED
	if (determinant < EPSILON) return NO_INTERSECTION;
#else
	if (determinant > -EPSILON || determinant < EPSILON) return 0.0f;
#endif

	float f = 1.0f / determinant;
	float4 s = rayPtr->origin - v0;

	float u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f) return NO_INTERSECTION;

	float4 q = cross(s, edge0);
	float v = f * dot(rayPtr->direction, q);
	if (v < 0.0f || u + v > 1.0f) return NO_INTERSECTION;

	float t = f * dot(edge1, q);
	return Intersection { t, trianglePtr->materialIdx, cross(edge0, edge1) };
}

__global__ void TraverseScene(Ray* rays, int rayCount, Triangle* triangles, int triangleCount, Vertex* vertices, Intersection* intersections) {
	uint rayIdx = threadIdx.x + blockIdx.x * blockDim.x;
	if (rayIdx >= rayCount || rays[rayIdx].direction == ZERO_VECTOR) return;

	for (int triangleIdx = 0; triangleIdx < triangleCount; triangleIdx++) {
		Intersection intersection = RayIntersectsTriangle(&rays[rayIdx], &triangles[triangleIdx], vertices);
		if (intersection.t > EPSILON && intersection.t < intersections[rayIdx].t) {
			intersections[rayIdx] = intersection;
		}
	}
}

__device__ float4 GetDiffuseReflection(float4 normal, curandStateXORWOW_t* rngStatePtr) {
	// Generate a random vector.
	// We want to make sure this random vector is not longer than 1, because we normalize this vector,
	// the resulting vector may skew towards the corners of the unit cube because of how we obtain this random vector.
	// Also ensure this loop does execture infinitely.
	// This still causes a slight skew towards the corners, so a better way to obtain a random unit vector is necessary.
	float4 result;
	uint loopCounter = 0;
	do {
		result = make_float4(curand_uniform(rngStatePtr), curand_uniform(rngStatePtr), curand_uniform(rngStatePtr), 0.0f);
		result = result * 2.0f - 1.0f;

		loopCounter++;
	} while (lengthSquared(result) > 1.0f && loopCounter <= 10);

	// Normalize the result and ensure it's pointing in the normal vector's hemisphere half.
	result = normalize(result);
	return dot(result, normal) > 0.0f ? result : -result;
}

__global__ void Intersect(Ray* rays, int rayCount, Intersection* intersections, Material* materials, curandStateXORWOW_t* rngStates, float4* frameBuffer) {
	uint rayIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (rayIdx >= rayCount || rays[rayIdx].direction == ZERO_VECTOR) return;

	if (!intersections[rayIdx].Hit()) {
		frameBuffer[rayIdx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
		return;
	}

	Material* materialPtr = &materials[intersections[rayIdx].materialIdx];
	float4 materialColor = materialPtr->color;

	if (materialPtr->type == Material::MaterialType::DIFFUSE) {
		float4 reflection = GetDiffuseReflection(intersections[rayIdx].normal, &rngStates[rayIdx]);
		rays[rayIdx].origin += rays[rayIdx].direction * intersections[rayIdx].t + reflection * EPSILON;
		rays[rayIdx].direction = reflection;

		frameBuffer[rayIdx] *= dot(intersections[rayIdx].normal, reflection) * 2.0f * materialColor;
	}
	else if (materialPtr->type == Material::MaterialType::EMISSIVE) {
		frameBuffer[rayIdx] *= materialColor;

		// Use this as "path finished" until compaction is implemented.
		rays[rayIdx].direction = ZERO_VECTOR;
	}
	else {
		frameBuffer[rayIdx] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);

		// Use this as "path finished" until compaction is implemented.
		rays[rayIdx].direction = ZERO_VECTOR;
	}
}
