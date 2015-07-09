#pragma once
#include "../gMF_define.h"
#include "../objects/gMF_BF_info.h"
#include "../objects/gMF_GF_info.h"

#include"../../gSLIC_Lib/gSLIC.h"

namespace gMF
{
class filter_engine
	{
    protected:

		Float3Image *reference_rgb_image;
		FloatArray *filter_data_in;
		FloatArray *filter_data_out;

        FloatArray *filter_data_tmp;

		Float3Image *subsample_ref_image;
		FloatArray *subsample_filter_data;
		FloatArray *subsample_filter_result;

        FloatArray *accum_data;
        FloatArray *accum_weight;

		Vector2i img_size;
		Vector2i subsample_size;

		int filter_data_dim;

		float sigma_rgb;

        bool gslic_segmented;
        gSLIC::objects::settings gslic_settings;
        gSLIC::engines::seg_engine_GPU *gslic_engine;
        FloatArray *gslic_accum_fdata;
        FloatArray *gslic_accum_weight;
        
    protected:
		
		void filter_bilateral_exact(float* out_data_ptr, float weight, int dim, int w, int h, BF_info* bf_info, bool additive);
        void filter_bilateral_approximate(float weight, int dim, int w, int h, BF_info* bf_info, bool additive,float* out_data_ptr=NULL);
        void filter_bilateral_splat_slice(float weight, int dim, int w, int h, BF_info* bf_info, bool additive,float* out_data_ptr=NULL);

        void filter_bilateral_superpixel(float weight, int dim, int w, int h, BF_info* bf_info, bool additive,float* out_data_ptr=NULL);
        
        void bilateral_filter_post_processing(BF_info* bf_info, float* out_data_ptr=NULL);

		void bilateral_subsample_filter_data(const Float3Image* ref_img, const FloatArray* in_data, FloatArray* out_data);
		void bilateral_subsample_reference_image(const Float3Image* in_img, Float3Image* out_img);

	public:

		filter_engine(int img_w, int img_h, int dim);
		~filter_engine();

		void load_reference_image(uchar* in_img_ptr, int w, int h);
		void load_filter_data(float* in_data_ptr, int dim, int w, int h);
		
        void filter_bilateral( float weight, int dim, int w, int h, BF_info* bf_info, bool additive,float* out_data_ptr=NULL);

		void filter_gaussian(float weight, int dim, int w, int h, GF_info* gf_info, bool additive, float* out_data_ptr=NULL );

		void test_subsample_filter(float* out_img, int w, int h, int dim, BF_info* bf_info);

	};



}

