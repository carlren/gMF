#include "gMF_fileter_engine.h"
#include "gMF_filter_engine_kernel.h"

#include <fstream>

using namespace std;
using namespace gMF;

filter_engine::filter_engine(int img_w, int img_h, int dim)
{
	img_size.x = img_w;
	img_size.y = img_h;

	filter_data_dim = dim;

	reference_rgb_image = new Float3Image(img_size, true, true);
	filter_data_in = new FloatArray(dim * img_w * img_h, true, true);
	filter_data_out = new FloatArray(dim * img_w * img_h, true, true);

    filter_data_tmp = new FloatArray(dim * img_w * img_h, true, true);

	// setup the sub-sampled data for passion sampling
	subsample_size = img_size / HIERARCHY_FACTOR;
	subsample_ref_image = new Float3Image(subsample_size, true, true);
	subsample_filter_data = new FloatArray(subsample_size.x * subsample_size.y* filter_data_dim, true, true);
	subsample_filter_result = new FloatArray(subsample_size.x * subsample_size.y* filter_data_dim, true, true);

    accum_data = new FloatArray(subsample_size.x * subsample_size.y * filter_data_dim, true, true);
    accum_weight = new FloatArray(subsample_size.x * subsample_size.y, true, true);

	sigma_rgb = -1.0;
}

gMF::filter_engine::~filter_engine()
{
	delete reference_rgb_image;
	delete filter_data_in;
	delete filter_data_out;
	
    delete filter_data_tmp;

	delete subsample_ref_image;
	delete subsample_filter_data;

    delete accum_data;
    delete accum_weight;
}

// ----------------------------------------------------
//
//	load data
//
// ----------------------------------------------------


// TODO: load into texture memory to speedup
void gMF::filter_engine::load_reference_image(uchar* in_img_ptr, int w, int h)
{
	Vector3f* rgb_ptr = reference_rgb_image->GetData(MEMORYDEVICE_CPU);

	for (int i = 0; i < w * h; i++)
	{
		rgb_ptr[i].r = in_img_ptr[i * 3];
		rgb_ptr[i].g = in_img_ptr[i * 3 + 1];
		rgb_ptr[i].b = in_img_ptr[i * 3 + 2];
	}

	reference_rgb_image->UpdateDeviceFromHost();
}


void gMF::filter_engine::load_filter_data(float* in_data_ptr, int dim, int w, int h)
{
	float* filter_data_ptr = filter_data_in->GetData(MEMORYDEVICE_CPU);
	memcpy(filter_data_ptr, in_data_ptr, w*h*dim*sizeof(float));
	filter_data_in->UpdateDeviceFromHost();
}


// ----------------------------------------------------
//
//	sub-sampling
//
// ----------------------------------------------------


void gMF::filter_engine::bilateral_subsample_reference_image(const Float3Image* in_img, Float3Image* out_img)
{
	const Vector3f *in_img_ptr = in_img->GetData(MEMORYDEVICE_CUDA);
	Vector3f *out_img_ptr = out_img->GetData(MEMORYDEVICE_CUDA);

	Vector2i size_old = in_img->noDims;
	Vector2i size_new = out_img->noDims;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)size_new.x / (float)blockSize.x), (int)ceil((float)size_new.y / (float)blockSize.y));

    bilateral_subsample_reference_image_device << <gridSize, blockSize >> >(
                                                                              in_img_ptr,
                                                                              out_img_ptr,
                                                                              sigma_rgb,
                                                                              size_old,
                                                                              size_new);
}


void gMF::filter_engine::bilateral_subsample_filter_data(const Float3Image* ref_img, const FloatArray* in_data, FloatArray* out_data)
{
	const Vector3f *in_img_ptr = ref_img->GetData(MEMORYDEVICE_CUDA);
	const float *in_data_ptr = in_data->GetData(MEMORYDEVICE_CUDA);
	float *out_data_ptr = out_data->GetData(MEMORYDEVICE_CUDA);

	Vector2i size_old = ref_img->noDims;
	Vector2i size_new = size_old / HIERARCHY_FACTOR;

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)size_new.x / (float)blockSize.x), (int)ceil((float)size_new.y / (float)blockSize.y));

    bilateral_subsample_filter_data_device << <gridSize, blockSize >> >(
                                                                          in_img_ptr,
                                                                          in_data_ptr,
                                                                          out_data_ptr,
                                                                          sigma_rgb,
                                                                          size_old,
                                                                          size_new,
                                                                          filter_data_dim);
}


// ----------------------------------------------------
//
//	filtering
//
// ----------------------------------------------------


void gMF::filter_engine::filter_bilateral(float weight, int dim, int w, int h, BF_info* bf_info, bool additive,float* out_data_ptr)
{


// option 1: use splat-slice with passion sampling
//    filter_bilateral_splat_slice(weight, dim, w, h, bf_info, additive);

// option 2: use direct passion sampling
      filter_bilateral_approximate(weight, dim, w, h, bf_info, additive);

    // post process the result
    bilateral_filter_post_processing(bf_info,out_data_ptr);

}

void gMF::filter_engine::filter_bilateral_exact(float* out_data_ptr, float weight, int dim, int w, int h, BF_info* bf_info, bool additive)
{
	float *fdata_in_ptr = filter_data_in->GetData(MEMORYDEVICE_CUDA);
	float *fdata_out_ptr = filter_data_out->GetData(MEMORYDEVICE_CUDA);
	Vector3f *ref_img_ptr = reference_rgb_image->GetData(MEMORYDEVICE_CUDA);

	float *bi_xy_table_ptr = bf_info->xy_lookup_table->GetData(MEMORYDEVICE_CUDA);
	float *bi_rgb_table_ptr = bf_info->rgb_lookup_table->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	// reset output data if not additive
	if (!additive) ORcudaSafeCall(cudaMemset(fdata_out_ptr, 0, img_size.x*img_size.y*filter_data_dim*sizeof(float)));

	filter_bilateral_exact_device << <gridSize, blockSize >> >(
		ref_img_ptr,
		fdata_in_ptr,
		fdata_out_ptr,
		img_size,
		bi_xy_table_ptr,
		bi_rgb_table_ptr,
		dim,
		bf_info->sigma_xy * 2,
		weight
		);

	filter_data_out->UpdateHostFromDevice();
	memcpy(out_data_ptr, filter_data_out->GetData(MEMORYDEVICE_CPU), w*h*dim*sizeof(float));
}

void gMF::filter_engine::filter_bilateral_approximate( float weight, int dim, int w, int h, BF_info* bf_info, bool additive, float* out_data_ptr)
{
	// subsample data by a factor of 2
	if (bf_info->sigma_rgb != sigma_rgb){
		sigma_rgb = bf_info->sigma_rgb;
		bilateral_subsample_reference_image(reference_rgb_image, subsample_ref_image);
	}
	bilateral_subsample_filter_data(reference_rgb_image, filter_data_in, subsample_filter_data);

	float *sub_fdata_in_ptr = subsample_filter_data->GetData(MEMORYDEVICE_CUDA);
	float *sub_fdata_out_ptr = subsample_filter_result->GetData(MEMORYDEVICE_CUDA);
	float *fdata_in_ptr = filter_data_in->GetData(MEMORYDEVICE_CUDA);
	float *fdata_out_ptr = filter_data_out->GetData(MEMORYDEVICE_CUDA);

	Vector3f *sub_img_ptr = subsample_ref_image->GetData(MEMORYDEVICE_CUDA);
	Vector3f *ref_img_ptr = reference_rgb_image->GetData(MEMORYDEVICE_CUDA);

	float *bi_xy_table_ptr = bf_info->xy_lookup_table->GetData(MEMORYDEVICE_CUDA);
	float *bi_rgb_table_ptr = bf_info->rgb_lookup_table->GetData(MEMORYDEVICE_CUDA);
	Vector2i* sample_ptr = bf_info->sample_array->GetData(MEMORYDEVICE_CUDA);


	// filter the subsampled data
	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)subsample_size.x / (float)blockSize.x), (int)ceil((float)subsample_size.y / (float)blockSize.y));

	filter_bilateral_possion_sample_device << <gridSize, blockSize >> >(
		sub_img_ptr,
		sub_fdata_in_ptr,
		sub_fdata_out_ptr,
		subsample_size,
		bi_xy_table_ptr,
		bi_rgb_table_ptr,
		sample_ptr,
		bf_info->no_samples,
		filter_data_dim
		);


	// reset output data if not additive
	if (!additive) ORcudaSafeCall(cudaMemset(fdata_out_ptr, 0, img_size.x*img_size.y*filter_data_dim*sizeof(float)));

	// now use multi-linear interpolation to write data back
	gridSize = dim3((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	multi_linear_interpolation_device << <gridSize, blockSize >> >(
		ref_img_ptr,
		sub_img_ptr,
		fdata_in_ptr,
		sub_fdata_out_ptr,
		fdata_out_ptr,
		img_size,
		subsample_size,
		bi_rgb_table_ptr,
		filter_data_dim,
		weight);

    if(out_data_ptr!=NULL){
        filter_data_out->UpdateHostFromDevice();
        memcpy(out_data_ptr, filter_data_out->GetData(MEMORYDEVICE_CPU), w*h*dim*sizeof(float));
    }
}

void gMF::filter_engine::filter_bilateral_splat_slice(float weight, int dim, int w, int h, BF_info *bf_info, bool additive, float* out_data_ptr)
{
    // subsample data by a factor of 2
    if (bf_info->sigma_rgb != sigma_rgb){
        sigma_rgb = bf_info->sigma_rgb;
        bilateral_subsample_reference_image(reference_rgb_image, subsample_ref_image);
    }
    bilateral_subsample_filter_data(reference_rgb_image, filter_data_in, subsample_filter_data);

    float *sub_fdata_in_ptr = subsample_filter_data->GetData(MEMORYDEVICE_CUDA);
    float *accum_data_ptr = accum_data->GetData(MEMORYDEVICE_CUDA);
    float *accum_weight_ptr = accum_weight->GetData(MEMORYDEVICE_CUDA);

    float *fdata_in_ptr = filter_data_in->GetData(MEMORYDEVICE_CUDA);
    float *fdata_out_ptr = filter_data_out->GetData(MEMORYDEVICE_CUDA);

    Vector3f *sub_img_ptr = subsample_ref_image->GetData(MEMORYDEVICE_CUDA);
    Vector3f *ref_img_ptr = reference_rgb_image->GetData(MEMORYDEVICE_CUDA);

    float *bi_xy_table_ptr = bf_info->xy_lookup_table->GetData(MEMORYDEVICE_CUDA);
    float *bi_rgb_table_ptr = bf_info->rgb_lookup_table->GetData(MEMORYDEVICE_CUDA);
    Vector2i* sample_ptr = bf_info->sample_array->GetData(MEMORYDEVICE_CUDA);

    ORcudaSafeCall(cudaMemset(accum_data_ptr, 0, subsample_size.x*subsample_size.y*filter_data_dim*sizeof(float)));
    ORcudaSafeCall(cudaMemset(accum_weight_ptr, 0, subsample_size.x*subsample_size.y*sizeof(float)));

    // filter the the subsampled data into accum_data and accum_weight
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((int)ceil((float)subsample_size.x / (float)blockSize.x), (int)ceil((float)subsample_size.y / (float)blockSize.y));

    filter_bilateral_possion_splat_device << <gridSize, blockSize >> >(
        sub_img_ptr,
        sub_fdata_in_ptr,
        accum_data_ptr,
        accum_weight_ptr,
        subsample_size,
        bi_xy_table_ptr,
        bi_rgb_table_ptr,
        sample_ptr,
        bf_info->no_samples,
        filter_data_dim
        );

    // reset output data if not additive
    if (!additive) ORcudaSafeCall(cudaMemset(fdata_out_ptr, 0, img_size.x*img_size.y*filter_data_dim*sizeof(float)));

    // now use multi-linear interpolation to slice data back
    gridSize = dim3((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

    multi_linear_slicing_device << <gridSize, blockSize >> >(
        ref_img_ptr,
        sub_img_ptr,
        fdata_in_ptr,
       accum_data_ptr,
       accum_weight_ptr,
        fdata_out_ptr,
        img_size,
        subsample_size,
        bi_rgb_table_ptr,
        filter_data_dim,
        weight);

    if(out_data_ptr!=NULL){
        filter_data_out->UpdateHostFromDevice();
        memcpy(out_data_ptr, filter_data_out->GetData(MEMORYDEVICE_CPU), w*h*dim*sizeof(float));
    }
}

void gMF::filter_engine::filter_gaussian(float* out_data_ptr, float weight, int dim, int w, int h, GF_info* gf_info, bool additive)
{
    float *fdata_in_ptr = filter_data_in->GetData(MEMORYDEVICE_CUDA);
    float *fdata_out_ptr = filter_data_out->GetData(MEMORYDEVICE_CUDA);

    float *xy_table_ptr = gf_info->xy_lookup_table->GetData(MEMORYDEVICE_CUDA);

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

    // reset output data if not additive
    if (!additive) ORcudaSafeCall(cudaMemset(fdata_out_ptr, 0, img_size.x*img_size.y*filter_data_dim*sizeof(float)));

    gaussian_filter_device << <gridSize, blockSize >> >(
        fdata_in_ptr,
        fdata_out_ptr,
        img_size,
        xy_table_ptr,
        gf_info->sigma_xy * 2,
        dim,
        weight
        );

    filter_data_out->UpdateHostFromDevice();
    memcpy(out_data_ptr, filter_data_out->GetData(MEMORYDEVICE_CPU), w*h*dim*sizeof(float));
}

void gMF::filter_engine::bilateral_filter_post_processing(BF_info *bf_info, float *out_data_ptr)
{
    filter_data_tmp->SetFrom(filter_data_out,FloatArray::CUDA_TO_CUDA);

    float *fdata_in_ptr = filter_data_in->GetData(MEMORYDEVICE_CUDA);
    float *fdata_out_ptr = filter_data_out->GetData(MEMORYDEVICE_CUDA);
    float *fdata_tmp_ptr = filter_data_tmp->GetData(MEMORYDEVICE_CUDA);

    Vector3f *ref_img_ptr = reference_rgb_image->GetData(MEMORYDEVICE_CUDA);

    float *bi_rgb_table_ptr = bf_info->rgb_lookup_table->GetData(MEMORYDEVICE_CUDA);

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

    bilateral_filter_post_processing_device << <gridSize, blockSize >> >(
        ref_img_ptr,
        fdata_in_ptr,
        fdata_tmp_ptr,
        fdata_out_ptr,
        img_size,
        bi_rgb_table_ptr,
        filter_data_dim
        );

    if(out_data_ptr!=NULL){
        filter_data_out->UpdateHostFromDevice();
        memcpy(out_data_ptr, filter_data_out->GetData(MEMORYDEVICE_CPU), filter_data_out->dataSize*sizeof(float));
    }

}

//-------------------------------
//
// testing function
//
//-------------------------------


void gMF::filter_engine::test_subsample_filter(float* out_img, int w, int h, int dim, BF_info* bf_info)
{
	// subsample data by a factor of 2
	if (bf_info->sigma_rgb != sigma_rgb){
		sigma_rgb = bf_info->sigma_rgb;
		bilateral_subsample_reference_image(reference_rgb_image, subsample_ref_image);
	}
	bilateral_subsample_filter_data(reference_rgb_image, filter_data_in, subsample_filter_data);

	float *sub_fdata_in_ptr = subsample_filter_data->GetData(MEMORYDEVICE_CUDA);
	float *sub_fdata_out_ptr = subsample_filter_result->GetData(MEMORYDEVICE_CUDA);
	float *fdata_out_ptr = filter_data_out->GetData(MEMORYDEVICE_CUDA);

	Vector3f *sub_img_ptr = subsample_ref_image->GetData(MEMORYDEVICE_CUDA);
	Vector3f *ref_img_ptr = reference_rgb_image->GetData(MEMORYDEVICE_CUDA);

	float *bi_xy_table_ptr = bf_info->xy_lookup_table->GetData(MEMORYDEVICE_CUDA);
	float *bi_rgb_table_ptr = bf_info->rgb_lookup_table->GetData(MEMORYDEVICE_CUDA);
	Vector2i* sample_ptr = bf_info->sample_array->GetData(MEMORYDEVICE_CUDA);


	// filter the subsampled data
	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)subsample_size.x / (float)blockSize.x), (int)ceil((float)subsample_size.y / (float)blockSize.y));

	filter_bilateral_possion_sample_device << <gridSize, blockSize >> >(
		sub_img_ptr,
		sub_fdata_in_ptr,
		sub_fdata_out_ptr,
		subsample_size,
		bi_xy_table_ptr,
		bi_rgb_table_ptr,
		sample_ptr,
		bf_info->no_samples,
		filter_data_dim
		);



	memset(out_img, 0, w*h*dim*sizeof(float));


	//subsample_ref_image->UpdateHostFromDevice();
	//Vector3f* ref_img_ptr = subsample_ref_image->GetData(MEMORYDEVICE_CPU);

	//for (int i = 0; i < subsample_size.y;i++){
	//	for (int j = 0; j < subsample_size.x;j++){
	//		out_img[(i*w + j)*dim + 0] = ref_img_ptr[i*subsample_size.x + j].r;
	//		out_img[(i*w + j)*dim + 1] = ref_img_ptr[i*subsample_size.x + j].g;
	//		out_img[(i*w + j)*dim + 2] = ref_img_ptr[i*subsample_size.x + j].b;
	//	}
	//}

	//subsample_filter_data->UpdateHostFromDevice();
	//float* sub_data_ptr = subsample_filter_data->GetData(MEMORYDEVICE_CPU);

	//for (int i = 0; i < subsample_size.y;i++){
	//	for (int j = 0; j < subsample_size.x;j++){
	//		for (int k = 0; k < dim;k++){
	//			out_img[(i*w + j)*dim + k] = sub_data_ptr[(i*subsample_size.x + j)*dim + k];
	//		}
	//	}
	//}

	subsample_filter_result->UpdateHostFromDevice();
	float* sub_data_ptr = subsample_filter_result->GetData(MEMORYDEVICE_CPU);

	for (int i = 0; i < subsample_size.y; i++){
		for (int j = 0; j < subsample_size.x; j++){
			for (int k = 0; k < dim; k++){
				out_img[(i*w + j)*dim + k] = sub_data_ptr[(i*subsample_size.x + j)*dim + k];
			}
		}
	}
}

