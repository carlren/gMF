#pragma once
#include "../objects/gMF_bf_grid.h"

namespace gMF
{
	class bf_grid_engine
	{
	private:

		bf_grid* bgrid;

		Float2Image* xy_data;
		Float3Image* rgb_data;

		float sigma_xy;
		float sigma_rgb;

		Vector2i img_size;
		int dim;

	private:

		void prepare_reference_img(const UChar3Image * in_img);

		void splat(const float* in_array, int dim);
		void blur();
		void slice(float* out_array);

	public:
		bf_grid_engine(Vector2i s_xy, Vector3i s_rgb, float sigma_xy, float sigma_rgb, int dim);
		~bf_grid_engine();

		void filter_distribution(const UChar3Image * in_img, const float* in_array, float* out_array, int dim);

	};
}