#pragma once

using namespace std;
using namespace gMF;


__global__ void prepare_reference_img_device(const Vector3u* in_img_ptr, Vector2f* out_xy_ptr, Vector3f* out_rgb_ptr, Vector2i image_size, float sigma_xy, float sigma_rgb)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > image_size.x - 1 || y > image_size.y - 1) return;

	int idx = y*image_size.x + y;
	Vector3u pix = in_img_ptr[idx];

	out_xy_ptr[idx].x = x / sigma_xy;
	out_xy_ptr[idx].y = y / sigma_xy;
	out_rgb_ptr[idx].r = pix.r / sigma_rgb;
	out_rgb_ptr[idx].g = pix.g / sigma_rgb;
	out_rgb_ptr[idx].b = pix.b / sigma_rgb;

}

_CPU_AND_GPU_CODE_ int pt2idx(int x, int y, int r, int g, int b, Vector2i size_xy, Vector3i size_rgb)
{
	return (((r * size_rgb.r + g)*size_rgb.g + b)*size_rgb.b + y)*size_xy.x + x;
}

__global__ void splat_device(const Vector2f* xy_ptr, const Vector3f* rgb_ptr, const float* in_array, float** grid_ptr, float* data_array_ptr, Vector2i image_size, Vector2i grid_size_xy, Vector3i grid_size_rgb, int dim)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > image_size.x - 1 || y > image_size.y - 1) return;

	int img_idx = y*image_size.x + y;

	Vector2f xy = xy_ptr[img_idx];
	Vector3f rgb = rgb_ptr[img_idx];

	int grid_idx = pt2idx((int)(xy.x + 0.5f), (int)(xy.y + 0.5f), (int)(rgb.r + 0.5f), (int)(rgb.g + 0.5f), (int)(rgb.b + 0.5f), grid_size_xy, grid_size_rgb);

	if (grid_ptr[grid_idx]==NULL)
	{
		atomicAdd(&gMF_last_avaiable_entry, 1);
		grid_ptr[grid_idx] = &data_array_ptr[gMF_last_avaiable_entry*(dim + 1)];
	}

	for (int i = 0; i < dim; i++) atomicAdd(&grid_ptr[grid_idx][i], in_array[img_idx*dim + i]);
	atomicAdd(&grid_ptr[grid_idx][dim], 1);
}

