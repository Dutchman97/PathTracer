#include <cuda_runtime.h>

struct Ray {
	float4 origin;
	float4 direction;
};

struct Vertex {
	float4 position;
};

struct Triangle {
	uint1 vertexIdx0;
	uint1 vertexIdx1;
	uint1 vertexIdx2;
};
