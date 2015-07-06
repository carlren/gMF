#pragma once
#include "../gMF_define.h"

namespace gMF
{
	class BF_info
	{
	public:

		float sigma_xy;
		float sigma_rgb;

		FloatArray* xy_lookup_table;
		FloatArray* rgb_lookup_table;
		Point2iArray* sample_array;
		int no_samples;

		BF_info(float s_xy, float s_rgb)
		{
			sigma_xy = s_xy;
			sigma_rgb = s_rgb;

			// pre-compute the gaussian look up table
			xy_lookup_table = new FloatArray(MAX_XY_SEARCH_RANGE, true, true);
			float* host_ptr = xy_lookup_table->GetData(MEMORYDEVICE_CPU);
			for (int i = 0; i < MAX_XY_SEARCH_RANGE; i++)
			{
				host_ptr[i] = expf(-(i * i) / (2 * sigma_xy * sigma_xy));
			}
			xy_lookup_table->UpdateDeviceFromHost();

			// pre-compute the rgb look up table
			rgb_lookup_table = new FloatArray(MAX_RGB_SEARCH_RANGE, true, true);
			host_ptr = rgb_lookup_table->GetData(MEMORYDEVICE_CPU);
			for (int i = 0; i < MAX_RGB_SEARCH_RANGE; i++)
			{
				host_ptr[i] = expf(-(i * i) / (2 * sigma_rgb * sigma_rgb));
			}
			rgb_lookup_table->UpdateDeviceFromHost();

			// compute the passion sampling grid
			sample_array = new Point2iArray(MAX_SAMPLE_TOTAL, true, true);

            int no_of_points_per_line = std::min((int)ceil(sqrt(sigma_xy * 2.5)), (int)MIN_SAMPLE_1D);
			int half_no_of_points = floor((float)no_of_points_per_line / 2);
			float sample_radius = sigma_xy / (float)half_no_of_points;

			Vector2i* samples = sample_array->GetData(MEMORYDEVICE_CPU);
			no_samples = 0;

			for (int y = -half_no_of_points; y <= half_no_of_points; y++){
				for (int x = -half_no_of_points; x <= half_no_of_points; x++, no_samples++){

					samples[no_samples].x = x * sample_radius + sample_radius * 0.5 * ((float)rand() / (float)RAND_MAX);
					samples[no_samples].y = y * sample_radius + sample_radius * 0.5 * ((float)rand() / (float)RAND_MAX);

				}
			}

			sample_array->UpdateDeviceFromHost();
		}

		~BF_info()
		{
			delete xy_lookup_table;
			delete rgb_lookup_table;
			delete sample_array;
		}

	};


}
