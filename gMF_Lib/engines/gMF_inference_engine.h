#pragma once

#include "../gMF_define.h"
#include "gMF_fileter_engine.h"

namespace gMF
{
    class inference_engine: public filter_engine
    {
    private:

        int data_size;

        FloatArray  *unary_potential;
        FloatArray *log_of_current_Q_distribution;
        FloatArray *compatibility_matrix;

        // current Q distribution is saved in the filtering data

    public:

        void load_unary_potential(float* in_data_ptr);
        void load_compatibility_function(float* in_model_ptr);

        void exp_and_normalize();
        void apply_compatibility_transform();
        void substract_update_from_unary_potential();

        void get_Q_distribution(float* out_data_ptr);

        inference_engine(int img_w, int img_h, int dim);
        ~inference_engine();


    };
}

