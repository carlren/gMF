#pragma once

#include <device_functions.h>
#include "../gMF_define.h"


using namespace std;
using namespace gMF;

__global__ void exp_and_normalize_device(
        const float *log_ptr,
        float *Q_dist_ptr,
        Vector2i image_size,
        int dim)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > image_size.x - 1 || y > image_size.y - 1) return;

    int idx = y * image_size.x + x;

    float exp_sum = 0.0f;
    for (int k = 0; k < dim; k++)
    {
        Q_dist_ptr[idx * dim + k] = __expf( log_ptr[idx * dim + k]);
        exp_sum += Q_dist_ptr[idx * dim + k];
    }

    for (int k = 0; k < dim; k++)
    {
        Q_dist_ptr[idx * dim + k] /= exp_sum;
    }

}

__global__ void apply_compatibility_func_device(
        float* filter_result_ptr,
        float* cmpt_func,
        Vector2i image_size,
        int dim)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > image_size.x - 1 || y > image_size.y - 1) return;

    int idx = y * image_size.x + x;
    float tmp_data[MAX_FILTER_DIM] = {0.0f};

    for(int i=0;i<dim;i++)
    {
        for(int k=0;k<dim;k++)
        {
            tmp_data[i]+=filter_result_ptr[idx*dim+k] * cmpt_func[i*dim+k];
        }
    }

    for (int k=0;k<dim;k++)
    {
        filter_result_ptr[idx*dim + k] = tmp_data[k];
    }
}

__global__ void substract_update_device(
        const float *update_ptr,
        const float *unary_ptr,
        float *log_ptr,
        Vector2i image_size,
        int dim)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > image_size.x - 1 || y > image_size.y - 1) return;

    int idx = y * image_size.x + x;

    for (int k=0;k<dim;k++)
    {
        log_ptr[idx*dim+k] =  unary_ptr[idx*dim + k] - update_ptr[idx*dim + k] ;
    }
}
