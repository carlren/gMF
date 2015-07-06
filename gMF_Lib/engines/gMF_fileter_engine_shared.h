#pragma once
#include <device_functions.h>

#include "../gMF_define.h"


//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
_CPU_AND_GPU_CODE_ inline float euclidean_dist(gMF::Vector3f a, gMF::Vector3f b, float d)
{
	float mod = (b.x - a.x) * (b.x - a.x) +
		(b.y - a.y) * (b.y - a.y) +
		(b.z - a.z) * (b.z - a.z);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
	return __expf(-mod / (2.f * d * d));
#else
	return expf(-mod / (2.f * d * d));
#endif
}

//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
_CPU_AND_GPU_CODE_ inline float euclidean_dist(gMF::Vector3u a, gMF::Vector3u b, float d)
{
	float mod = (b.x - a.x) * (b.x - a.x) +
		(b.y - a.y) * (b.y - a.y) +
		(b.z - a.z) * (b.z - a.z);

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
	return __expf(-mod / (2.f * d * d));
#else
	return expf(-mod / (2.f * d * d));
#endif
}

