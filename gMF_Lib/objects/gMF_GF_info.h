#pragma once
#include "../gMF_define.h"

namespace gMF
{
	class GF_info
	{
	public:

		float sigma_xy;
		FloatArray* xy_lookup_table;
		
		GF_info(float s_xy)
		{
			sigma_xy = s_xy;

			// pre-compute the gaussian look up table
			xy_lookup_table = new FloatArray(MAX_XY_SEARCH_RANGE, true, true);
			float* host_ptr = xy_lookup_table->GetData(MEMORYDEVICE_CPU);
			for (int i = 0; i < MAX_XY_SEARCH_RANGE; i++)
			{
				host_ptr[i] = expf(-(i * i) / (2 * sigma_xy * sigma_xy));
			}
			xy_lookup_table->UpdateDeviceFromHost();
		}

		~GF_info()
		{
			delete xy_lookup_table;
		}

	};


}