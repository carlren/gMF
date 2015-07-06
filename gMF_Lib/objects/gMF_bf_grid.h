#pragma once
#include "../gMF_define.h"


__device__ int gMF_last_avaiable_entry;

namespace gMF
{
	class bf_grid
	{
	public:

		ORUtils::MemoryBlock<float*>* grid;
		FloatArray* data_array;

		Vector2i size_xy;
		Vector3i size_rgb;
		int dim;

	public:

		bf_grid(Vector2i sz_xy, Vector3i sz_rgb, float sigma_xy, float sigma_rgb, int dim)
		{
			size_xy.x = ceil(sz_xy.x / sigma_xy);
			size_xy.y = ceil(sz_xy.y / sigma_xy);

			size_rgb.r = ceil(sz_rgb.r / sigma_rgb);
			size_rgb.g = ceil(sz_rgb.g / sigma_rgb);
			size_rgb.b = ceil(sz_rgb.b / sigma_rgb);

			grid = new ORUtils::MemoryBlock<float*>(size_xy.x*size_xy.y*size_rgb.r*size_rgb.g*size_rgb.b, true, true);

			data_array = new FloatArray(sz_xy.x * sz_xy.y * (dim + 1), true, true);
		}

		~bf_grid()
		{
			delete grid;
			delete data_array;
		}

		void set_zero()
		{
			ORcudaSafeCall(cudaMemset(grid->GetData(MEMORYDEVICE_CUDA), NULL, grid->dataSize*sizeof(float*)));
			ORcudaSafeCall(cudaMemset(data_array->GetData(MEMORYDEVICE_CUDA), 0, data_array->dataSize*sizeof(float)));
			ORcudaSafeCall(cudaMemset(&gMF_last_avaiable_entry, 0, sizeof(int)));
		}

	};
}