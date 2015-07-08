#pragma once
#include "gMF_fileter_engine_shared.h"
#include "../../gSLIC_Lib/objects/gSLIC_spixel_info.h"

// safe to include namespace here, since it's only included in the .cu file
using namespace std;
using namespace gMF;

static const _CPU_AND_GPU_CONSTANT_ unsigned char gaussian3[3] = { 1, 2, 1 };

// ----------------------------------------------------
//
//	filtering using superpixel
//
// ----------------------------------------------------

__global__ void accum_filter_data_in_spixel_device(
        const gSLIC::objects::spixel_info*  spixel_center,
        const float         *fdata_in_ptr,
        const int             *gslic_idx_ptr,
        float                     *gslic_accum_data_ptr,
        float                     *gslic_accum_weight_ptr,
        Vector2i               slic_map_size,
        Vector2i               image_size,
        int                          radius,
        int                          dim
        )
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > slic_map_size.x - 1 || y > slic_map_size.y - 1) return;
    
    int spixel_idx = y * slic_map_size.x + x;
    const gSLIC::objects::spixel_info& spixel_c = spixel_center[spixel_idx];
  
    
	float data_sum[MAX_FILTER_DIM] = { 0.0f };
    float weight_sum = 0;
    
    for (int i=-radius; i<=radius; i++)
        for (int j=-radius; j<=radius; j++)
        {
            int ii = spixel_c.center.y + i;
			int jj = spixel_c.center.x + j;

			if (ii < 0 || ii >image_size.y - 1 || jj <0 || jj>image_size.x - 1) continue;
            
            int img_idx = ii * image_size.x + jj;
            if (gslic_idx_ptr[img_idx]!=spixel_idx) continue;
            
            for (int k = 0; k < dim; k++)
				data_sum[k] += fdata_in_ptr[img_idx*dim + k];
            weight_sum++;
        }
        
    for (int k = 0; k < dim; k++) gslic_accum_data_ptr[spixel_idx*dim + k] = data_sum[k];
    gslic_accum_weight_ptr[spixel_idx] = weight_sum;
    
}

__global__ void filter_bilateral_superpixel_device(
	const gSLIC::objects::spixel_info*  spixel_center,
    const float          *gslic_accum_data_ptr,
    const float          *gslic_weight_ptr,
    const int              *gslic_idx_ptr,    
    const Vector3f		*in_img_ptr,   
	const float			*fdata_in_ptr, 
	float                       *fdata_out_ptr, 
    Vector2i               slic_map_size,
	Vector2i			image_size, 
	float				*gaussian_table_ptr, 
	float				*euclid_table_ptr, 
	int					dim,
	int					radius,
	float				weight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > image_size.x - 1 || y > image_size.y - 1) return;

	int idx_center = y * image_size.x + x;

	float factor_sum = 0.0f;
	float data_sum[MAX_FILTER_DIM] = { 0.0f };

	const Vector3f& pixel_c = in_img_ptr[idx_center];
    const int spixel_idx = gslic_idx_ptr[idx_center];
    
    const int xs = spixel_idx % slic_map_size.x;
    const int ys = spixel_idx / slic_map_size.x;
   
    // adding original data in there
    factor_sum=gaussian_table_ptr[0]
            *gaussian_table_ptr[0]
            *euclid_table_ptr[0]
            *euclid_table_ptr[0]
            *euclid_table_ptr[0];
    
    for (int k = 0; k < dim; k++) data_sum[k] += fdata_in_ptr[(idx_center)*dim + k] * factor_sum;
    
    
	for (int i = -radius; i <= radius; i++)
		for (int j = -radius; j <= radius; j++)
		{
			int ii = ys + i;
			int jj = xs + j;

			if (ii < 0 || ii >slic_map_size.y - 1 || jj <0 || jj>slic_map_size.x - 1)
				continue;

            const gSLIC::objects::spixel_info& spixel_s = spixel_center[ii * slic_map_size.x + jj];

			float factor = gaussian_table_ptr[(int)abs(x - spixel_s.center.x)]
				* gaussian_table_ptr[(int)abs(y - spixel_s.center.y)]
				* euclid_table_ptr[(int)abs(pixel_c.r - spixel_s.color_info.r)]
				* euclid_table_ptr[(int)abs(pixel_c.g - spixel_s.color_info.g)]
				* euclid_table_ptr[(int)abs(pixel_c.b - spixel_s.color_info.b)];
            
			factor_sum += factor * gslic_weight_ptr[ii * slic_map_size.x + jj];
			for (int k = 0; k < dim; k++)
				data_sum[k] += gslic_accum_data_ptr[(ii * slic_map_size.x + jj)*dim + k] * factor;
		}

		for (int k = 0; k < dim; k++)
			fdata_out_ptr[idx_center*dim + k] += weight * data_sum[k] / factor_sum;        
}




// ----------------------------------------------------
//
//	filtering using splat-slice
//
// ----------------------------------------------------

__global__ void filter_bilateral_possion_splat_device(
        const Vector3f		*in_img_ptr,
        const float			*fdata_in_ptr,
        float                       *accum_data_ptr,
        float                       *accum_weight_ptr,
        Vector2i                image_size,
        float                       *gaussian_table_ptr,
        float                       *euclid_table_ptr,
        Vector2i                *sample_list,
        int                          no_samples,
        int                         dim)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > image_size.x - 1 || y > image_size.y - 1) return;

    int idx_center = y * image_size.x + x;

    float factor_sum = 0.0f;
    float data_sum[MAX_FILTER_DIM] = { 0.0f };

    const Vector3f& pixel_c = in_img_ptr[idx_center];

    for (int i = 0; i < no_samples; i++)
    {
        Vector2i  sample = sample_list[i];
        int jj = sample.x + x;
        int ii = sample.y + y;

        if (ii < 0 || ii >image_size.y - 1 || jj <0 || jj>image_size.x - 1)
            continue;

        const Vector3f& pixel_s = in_img_ptr[ii * image_size.x + jj];

        float factor = gaussian_table_ptr[abs(sample.x * HIERARCHY_FACTOR)]
            * gaussian_table_ptr[abs(sample.y * HIERARCHY_FACTOR)]
            * euclid_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
            * euclid_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
            * euclid_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

        factor_sum += factor;
        for (int k = 0; k < dim; k++)
            data_sum[k] += fdata_in_ptr[(ii * image_size.x + jj)*dim + k] * factor;

    }

    accum_weight_ptr[idx_center] = factor_sum;
    for (int k = 0; k < dim; k++)
        accum_data_ptr[idx_center*dim + k] = data_sum[k];
}

__global__ void multi_linear_slicing_device(
        const Vector3f      *in_img_ptr,
        const Vector3f      *sub_img_ptr,
        const float             *in_fdata_ptr,
        const float             *accum_data_ptr,
        const float             *accum_weight_ptr,
        float				*fdata_out_ptr,
        Vector2i			ori_img_size,
        Vector2i			sub_size,
        float				*rgb_table_ptr,
        int					dim,
        float				weight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > ori_img_size.x - 1 || y > ori_img_size.y - 1) return;

    int ori_idx = y * ori_img_size.x + x;

    float factor_sum = 0.0f;
    float data_sum[MAX_FILTER_DIM] = { 0.0f };

    const Vector3f& pixel_c = in_img_ptr[ori_idx];

    float xf_sub = (float) x / HIERARCHY_FACTOR;
    float yf_sub = (float) y / HIERARCHY_FACTOR;

    int x_sub = floor(xf_sub);
    int y_sub = floor(yf_sub);

    // (0,0)
    if (x_sub >= 0 && y_sub >=0 && x_sub < sub_size.x && y_sub < sub_size.y){
        int sub_idx = y_sub * sub_size.x + x_sub;
        const Vector3f& pixel_s = sub_img_ptr[sub_idx];

        float factor = (1 - (xf_sub - x_sub))
                    * rgb_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                    * rgb_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                    * rgb_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

        for (int k = 0; k < dim; k++)
            data_sum[k] += accum_data_ptr[sub_idx*dim + k] * factor;
        factor_sum += factor * accum_weight_ptr[sub_idx];
    }

    //(1,0)
    x_sub += 1;
    if (x_sub >= 0 && y_sub >= 0 && x_sub < sub_size.x && y_sub < sub_size.y){
        int sub_idx = y_sub * sub_size.x + x_sub;
        const Vector3f& pixel_s = sub_img_ptr[sub_idx];

        float factor = (x_sub - xf_sub)
                * rgb_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                * rgb_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                * rgb_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

    for (int k = 0; k < dim; k++)
        data_sum[k] += accum_data_ptr[sub_idx*dim + k] * factor;
    factor_sum += factor * accum_weight_ptr[sub_idx];
    }

    //(1,1)
    y_sub += 1;
    if (x_sub >= 0 && y_sub >= 0 && x_sub < sub_size.x && y_sub < sub_size.y){
        int sub_idx = y_sub * sub_size.x + x_sub;
        const Vector3f& pixel_s = sub_img_ptr[sub_idx];

        float factor = (y_sub - yf_sub)
                * rgb_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                * rgb_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                * rgb_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

    for (int k = 0; k < dim; k++)
        data_sum[k] += accum_data_ptr[sub_idx*dim + k] * factor;
    factor_sum += factor * accum_weight_ptr[sub_idx];
    }

    //(0,1)
    x_sub -= 1;
    if (x_sub >= 0 && y_sub >= 0 && x_sub < sub_size.x && y_sub < sub_size.y){
        int sub_idx = y_sub * sub_size.x + x_sub;
        const Vector3f& pixel_s = sub_img_ptr[sub_idx];

        float factor =  (1 - (yf_sub - y_sub))
                * rgb_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                * rgb_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                * rgb_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

    for (int k = 0; k < dim; k++)
        data_sum[k] += accum_data_ptr[sub_idx*dim + k] * factor;
    factor_sum += factor * accum_weight_ptr[sub_idx];
    }

    // add the original data
//    for (int k = 0; k < dim; k++)
//        data_sum[k] +=  in_fdata_ptr[ori_idx*dim + k] * 0.01f;
//    factor_sum += 0.01f;

    // write back to final result
    if (factor_sum > 10e-10)
    {
        for (int k = 0; k < dim; k++)
            fdata_out_ptr[ori_idx*dim + k] += weight * data_sum[k] / factor_sum;
    }
}


// ----------------------------------------------------
//
//	filtering using passion sampling
//
// ----------------------------------------------------


__global__ void filter_bilateral_possion_sample_device(
	const Vector3f		*in_img_ptr, 
	const float			*fdata_in_ptr, 
	float				*fdata_out_ptr, 
	Vector2i			image_size, 
	float				*gaussian_table_ptr, 
	float				*euclid_table_ptr, 
	Vector2i			*sample_list, 
	int					no_samples, 
	int					dim)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > image_size.x - 1 || y > image_size.y - 1) return;

	int idx_center = y * image_size.x + x;

	float factor_sum = 0.0f;
	float data_sum[MAX_FILTER_DIM] = { 0.0f };

	const Vector3f& pixel_c = in_img_ptr[idx_center];

	for (int i = 0; i < no_samples; i++)
	{
		Vector2i  sample = sample_list[i];
		int jj = sample.x + x;
		int ii = sample.y + y;

		if (ii < 0 || ii >image_size.y - 1 || jj <0 || jj>image_size.x - 1)
			continue;

		const Vector3f& pixel_s = in_img_ptr[ii * image_size.x + jj];

		float factor = gaussian_table_ptr[abs(sample.x * HIERARCHY_FACTOR)]
			* gaussian_table_ptr[abs(sample.y * HIERARCHY_FACTOR)]
			* euclid_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
			* euclid_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
			* euclid_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

		factor_sum += factor;
		for (int k = 0; k < dim; k++)
			data_sum[k] += fdata_in_ptr[(ii * image_size.x + jj)*dim + k] * factor;

	}

	if (factor_sum > 10e-10)
	{
		for (int k = 0; k < dim; k++)
			fdata_out_ptr[idx_center*dim + k] = data_sum[k] / factor_sum;
	}
}


__global__ void multi_linear_interpolation_device(
	const Vector3f		*in_img_ptr,
	const Vector3f		*sub_img_ptr,
	const float			*in_fdata_ptr,
	const float			*sub_fdata_ptr,
	float				*fdata_out_ptr,
	Vector2i			ori_img_size,
	Vector2i			sub_size,
	float				*rgb_table_ptr,
	int					dim,
	float				weight
	)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > ori_img_size.x - 1 || y > ori_img_size.y - 1) return;

	int ori_idx = y * ori_img_size.x + x;

	float factor_sum = 0.0f;
	float data_sum[MAX_FILTER_DIM] = { 0.0f };

	const Vector3f& pixel_c = in_img_ptr[ori_idx];

    float xf_sub = (float) x / HIERARCHY_FACTOR;
    float yf_sub = (float) y / HIERARCHY_FACTOR;

    int x_sub = floor(xf_sub);
    int y_sub = floor(yf_sub);

	// (0,0)
	if (x_sub >= 0 && y_sub >=0 && x_sub < sub_size.x && y_sub < sub_size.y){
		int sub_idx = y_sub * sub_size.x + x_sub;
		const Vector3f& pixel_s = sub_img_ptr[sub_idx];

        float factor = (1 - (xf_sub - x_sub))
                    * rgb_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                    * rgb_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                    * rgb_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];
		
		for (int k = 0; k < dim; k++) 
			data_sum[k] += sub_fdata_ptr[sub_idx*dim + k] * factor;
		factor_sum += factor;
	}

	//(1,0)
	x_sub += 1;
	if (x_sub >= 0 && y_sub >= 0 && x_sub < sub_size.x && y_sub < sub_size.y){
		int sub_idx = y_sub * sub_size.x + x_sub;
		const Vector3f& pixel_s = sub_img_ptr[sub_idx];

        float factor = (x_sub - xf_sub)
                    * rgb_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                    * rgb_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                    * rgb_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

		for (int k = 0; k < dim; k++)
			data_sum[k] += sub_fdata_ptr[sub_idx*dim + k] * factor;
		factor_sum += factor;
	}

	//(1,1)
	y_sub += 1;
	if (x_sub >= 0 && y_sub >= 0 && x_sub < sub_size.x && y_sub < sub_size.y){
		int sub_idx = y_sub * sub_size.x + x_sub;
		const Vector3f& pixel_s = sub_img_ptr[sub_idx];
		
        float factor = (y_sub - yf_sub)
                    * rgb_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                    * rgb_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                    * rgb_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

		for (int k = 0; k < dim; k++)
			data_sum[k] += sub_fdata_ptr[sub_idx*dim + k] * factor;
		factor_sum += factor;
	}

	//(0,1)
	x_sub -= 1;
	if (x_sub >= 0 && y_sub >= 0 && x_sub < sub_size.x && y_sub < sub_size.y){
		int sub_idx = y_sub * sub_size.x + x_sub;
		const Vector3f& pixel_s = sub_img_ptr[sub_idx];

        float factor =  (1 - (yf_sub - y_sub))
                    * rgb_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                    * rgb_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                    * rgb_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

		for (int k = 0; k < dim; k++)
			data_sum[k] += sub_fdata_ptr[sub_idx*dim + k] * factor;
		factor_sum += factor;
	}

	// write back to final result
	if (factor_sum > 10e-10)
	{
		for (int k = 0; k < dim; k++)
			fdata_out_ptr[ori_idx*dim + k] += weight * data_sum[k] / factor_sum;
	}
}


__global__ void gaussian_filter_device(
	const float			*fdata_in_ptr, 
	float				*fdata_out_ptr, 
	Vector2i			image_size, 
	float				*gaussian_table_ptr, 
	float				radius, 
	int					dim, 
	float				weight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > image_size.x - 1 || y > image_size.y - 1) return;

	int idx_center = y * image_size.x + x;

	float factor_sum = 0.0f;
	float data_sum[MAX_FILTER_DIM] = { 0.0f };

	for (int i = -radius; i < radius; i++)
		for (int j = -radius; j < radius; j++)
		{
			int ii = y + i;
			int jj = x + j;

			if (ii < 0 || ii >image_size.y - 1 || jj <0 || jj>image_size.x - 1)
				continue;

			float factor = gaussian_table_ptr[abs(i)] * gaussian_table_ptr[abs(j)];

			factor_sum += factor;
			for (int k = 0; k < dim; k++)
				data_sum[k] += fdata_in_ptr[(ii * image_size.x + jj)*dim + k] * factor;

		}

	if (factor_sum > 10e-10)
	{
		for (int k = 0; k < dim; k++)
			fdata_out_ptr[idx_center*dim + k] += weight * data_sum[k] / factor_sum;
	}
}

__global__ void filter_bilateral_exact_device(
	const Vector3f		*in_img_ptr, 
	const float			*fdata_in_ptr, 
	float				*fdata_out_ptr, 
	Vector2i			image_size, 
	float				*gaussian_table_ptr, 
	float				*euclid_table_ptr, 
	int					dim,
	int					radius,
	float				weight)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > image_size.x - 1 || y > image_size.y - 1) return;

	int idx_center = y * image_size.x + x;

	float factor_sum = 0.0f;
	float data_sum[MAX_FILTER_DIM] = { 0.0f };

	const Vector3f& pixel_c = in_img_ptr[idx_center];

	for (int i = -radius; i < radius; i++)
		for (int j = -radius; j < radius; j++)
		{
			int ii = y + i;
			int jj = x + j;

			if (ii < 0 || ii >image_size.y - 1 || jj <0 || jj>image_size.x - 1)
				continue;

			const Vector3f& pixel_s = in_img_ptr[ii * image_size.x + jj];

			float factor = gaussian_table_ptr[abs(i)]
				* gaussian_table_ptr[abs(j)]
				* euclid_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
				* euclid_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
				* euclid_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

			factor_sum += factor;
			for (int k = 0; k < dim; k++)
				data_sum[k] += fdata_in_ptr[(ii * image_size.x + jj)*dim + k] * factor;

		}

	if (factor_sum > 10e-10)
	{
		for (int k = 0; k < dim; k++)
			fdata_out_ptr[idx_center*dim + k] += weight * data_sum[k] / factor_sum;
	}
}


// ----------------------------------------------------
//
//	subsampling
//
// ----------------------------------------------------


__global__ void bilateral_subsample_reference_image_device(
        const Vector3f *in_img_ptr,
        Vector3f *out_img_ptr,
        float sigma_rgb,
        Vector2i old_img_size,
        Vector2i new_img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > new_img_size.x - 1 || y > new_img_size.y - 1) return;

	int x_old = x * HIERARCHY_FACTOR + HIERARCHY_FACTOR / 2;
	int y_old = y * HIERARCHY_FACTOR + HIERARCHY_FACTOR / 2;

	int idx_old = y_old * old_img_size.x + x_old;

	Vector3f c_pixel = in_img_ptr[idx_old];
	Vector3f rgb_sum(0, 0, 0);
	float factor, factor_sum = 0;

	/*3x3 kernel*/
	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
		{
			int ii = y_old + i;
			int jj = x_old + j;

			if (ii < 0 || ii >old_img_size.y - 1 || jj <0 || jj>old_img_size.x - 1)
				continue;

			Vector3f cur_pix = in_img_ptr[ii * old_img_size.x + jj];
            factor = euclidean_dist(cur_pix, c_pixel, sigma_rgb) * gaussian3[i + 1] * gaussian3[j + 1];
			rgb_sum += factor * cur_pix;
			factor_sum += factor;
		}

	out_img_ptr[y*new_img_size.x + x] = rgb_sum / factor_sum;
}


__global__ void bilateral_subsample_filter_data_device(
        const Vector3f *in_img_ptr,
        const float *in_data_ptr,
        float *out_data_ptr,
        float sigma_rgb,
        Vector2i old_img_size,
        Vector2i new_img_size,
        int dim)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > new_img_size.x - 1 || y > new_img_size.y - 1) return;

	int x_old = x * HIERARCHY_FACTOR + HIERARCHY_FACTOR / 2;
	int y_old = y * HIERARCHY_FACTOR + HIERARCHY_FACTOR / 2;

	Vector3f c_pixel = in_img_ptr[y_old * old_img_size.x + x_old];

	float data_sum[MAX_FILTER_DIM] = { 0.0f };

	float factor, factor_sum = 0;

	/*3x3 kernel*/
	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
		{
			int ii = y_old + i;
			int jj = x_old + j;

			if (ii < 0 || ii >old_img_size.y - 1 || jj <0 || jj>old_img_size.x - 1)
				continue;

			Vector3f cur_pix = in_img_ptr[ii * old_img_size.x + jj];
			factor = euclidean_dist(cur_pix, c_pixel, sigma_rgb) * gaussian3[i + 1] * gaussian3[j + 1];

			for (int k = 0; k < dim; k++) data_sum[k] += in_data_ptr[(ii * old_img_size.x + jj) * dim + k] * factor;
			factor_sum += factor;
		}

	for (int k = 0; k < dim; k++) out_data_ptr[(y*new_img_size.x + x)* dim + k] = data_sum[k] / factor_sum;
}


// ----------------------------------------------------
//
//	post processing
//
// ----------------------------------------------------


__global__ void bilateral_filter_post_processing_device(
    const Vector3f		*in_img_ptr,
    const float			*fdata_in_ptr,
    const float           *fdata_tmp_ptr,
    float				*fdata_out_ptr,
    Vector2i			image_size,
    float				*euclid_table_ptr,
    int					dim)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > image_size.x - 1 || y > image_size.y - 1) return;

    int idx_center = y * image_size.x + x;

    float factor_sum = 0.0f;
    float data_sum[MAX_FILTER_DIM] = { 0.0f };

    const Vector3f& pixel_c = in_img_ptr[idx_center];

    // 3x3 kernal
    for (int i = -1; i <= 1; i++)
        for (int j = -1; j <= 1; j++)
        {
            int ii = y + i;
            int jj = x + j;

            if (ii < 0 || ii >image_size.y - 1 || jj <0 || jj>image_size.x - 1)
                continue;

            const Vector3f& pixel_s = in_img_ptr[ii * image_size.x + jj];

            float factor = gaussian3[i+1] * gaussian3[j+1]
                * euclid_table_ptr[(int)abs(pixel_s.r - pixel_c.r)]
                * euclid_table_ptr[(int)abs(pixel_s.g - pixel_c.g)]
                * euclid_table_ptr[(int)abs(pixel_s.b - pixel_c.b)];

            factor_sum += factor;
            for (int k = 0; k < dim; k++)
                data_sum[k] += fdata_tmp_ptr[(ii * image_size.x + jj)*dim + k] * factor;

        }

    for (int k = 0; k < dim; k++)
    {
        data_sum[k] += fdata_in_ptr[idx_center*dim + k];
        factor_sum += euclid_table_ptr[0]*euclid_table_ptr[0]*euclid_table_ptr[0];

         fdata_out_ptr[idx_center*dim + k] =   data_sum[k] / factor_sum;
    }

}

