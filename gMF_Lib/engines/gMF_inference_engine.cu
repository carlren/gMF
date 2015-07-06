#include "gMF_inference_engine.h"
#include "gMF_inference_engine_kernel.h"

#include<iostream>

using namespace gMF;

inference_engine::inference_engine(int img_w, int img_h, int dim) : filter_engine(img_w,img_h,dim)
{
    data_size = img_w * img_h * dim;
    unary_potential = new FloatArray(img_w*img_h*dim,true,true);
    log_of_current_Q_distribution = new FloatArray(img_w*img_h*dim,true,true);
    compatibility_matrix = new FloatArray(dim*dim,true,true);
}

inference_engine::~inference_engine()
{
    delete unary_potential;
    delete log_of_current_Q_distribution;
    delete compatibility_matrix;
}


void inference_engine::load_unary_potential(float *in_data_ptr)
{
    float* unary_ptr = unary_potential->GetData(MEMORYDEVICE_CPU);
    memcpy(unary_ptr,in_data_ptr, data_size*sizeof(float));
    unary_potential->UpdateDeviceFromHost();
    log_of_current_Q_distribution->SetFrom(unary_potential, FloatArray::CUDA_TO_CUDA);
}

void inference_engine::load_compatibility_function(float *in_model_ptr)
{
    float* compatibility_ptr = compatibility_matrix->GetData(MEMORYDEVICE_CPU);
    memcpy(compatibility_ptr, in_model_ptr,filter_data_dim*filter_data_dim*sizeof(float));
    compatibility_matrix->UpdateDeviceFromHost();
}

void inference_engine::exp_and_normalize()
{
    float* log_ptr = log_of_current_Q_distribution->GetData(MEMORYDEVICE_CUDA);
    float* Q_dist_ptr = filter_data_in->GetData(MEMORYDEVICE_CUDA);

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

    exp_and_normalize_device<<<gridSize,blockSize>>>(
                log_ptr,
                Q_dist_ptr,
                img_size,
                filter_data_dim);

}

void inference_engine::apply_compatibility_transform()
{
    float* filter_result_ptr = filter_data_out->GetData(MEMORYDEVICE_CUDA);
    float* cmpt_func = compatibility_matrix->GetData(MEMORYDEVICE_CUDA);

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

    apply_compatibility_func_device<<<gridSize,blockSize>>>(
                filter_result_ptr,
                cmpt_func,
                img_size,
                filter_data_dim);
}

void inference_engine::substract_update_from_unary_potential()
{
    float *update_ptr = filter_data_out->GetData(MEMORYDEVICE_CUDA);
    float *unary_ptr = unary_potential->GetData(MEMORYDEVICE_CUDA);
    float *log_ptr = log_of_current_Q_distribution->GetData(MEMORYDEVICE_CUDA);

    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

    substract_update_device<<<gridSize,blockSize>>>(
                update_ptr,
                unary_ptr,
                log_ptr,
                img_size,
                filter_data_dim);

}


void inference_engine::get_Q_distribution(float *out_data_ptr)
{
    filter_data_in->UpdateHostFromDevice();
    float *cpu_data_ptr = filter_data_in->GetData(MEMORYDEVICE_CPU);
    memcpy(out_data_ptr,cpu_data_ptr,data_size*sizeof(float));
}
