#pragma once
#include "gMF_bf_grid_engine.h"
#include "gMF_bf_grid_engine_kernel.h"

#include <fstream>

using namespace std;
using namespace gMF;


gMF::bf_grid_engine::bf_grid_engine(Vector2i size_xy, Vector3i size_rgb, float sigma_xy, float sigma_rgb, int dim)
{
	bgrid = new bf_grid(size_xy, size_rgb, sigma_xy, sigma_rgb, dim);
	xy_data = new Float2Image(size_xy, true, true);
	rgb_data = new Float3Image(size_xy, true, true);

	this->img_size = size_xy;;
	this->dim = dim;
	this->sigma_xy = sigma_xy;
	this->sigma_rgb = sigma_rgb;
}

gMF::bf_grid_engine::~bf_grid_engine()
{
	delete bgrid;
}

void gMF::bf_grid_engine::filter_distribution(const UChar3Image* in_img, const float* in_array, float* out_array, int dim)
{
	prepare_reference_img(in_img);
	splat(in_array, dim);
	blur();
	slice(out_array);
}

void gMF::bf_grid_engine::splat(const float* in_array, int dim)
{
	bgrid->set_zero();

	float** grid_ptr = bgrid->grid->GetData(MEMORYDEVICE_CUDA);
	float* data_array_ptr = bgrid->data_array->GetData(MEMORYDEVICE_CUDA);

	//dim3 blockSize(bgrid->size_xy.x, bgrid->size_xy.y);
	//dim3 gridSize(bgrid->size_rgb.r, bgrid->size_rgb.g, bgrid->size_rgb.b);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	
}

void gMF::bf_grid_engine::blur()
{

}

void gMF::bf_grid_engine::slice(float* out_array)
{

}

void gMF::bf_grid_engine::prepare_reference_img(const UChar3Image * in_img)
{
	const Vector3u* in_img_ptr = in_img->GetData(MEMORYDEVICE_CUDA);
	Vector2f* out_xy_ptr = xy_data->GetData(MEMORYDEVICE_CUDA);
	Vector3f* out_rgb_ptr = rgb_data->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	prepare_reference_img_device<<<gridSize,blockSize>>>(in_img_ptr, out_xy_ptr, out_rgb_ptr, img_size, sigma_xy, sigma_rgb);
}
