#pragma once
#include "../gSLIC_defines.h"
#include "../objects/gSLIC_settings.h"
#include "../objects/gSLIC_spixel_info.h"

namespace gSLIC
{
	namespace engines
	{
		class seg_engine
		{
		protected:

			// normalizing distances
			float max_color_dist;
			float max_xy_dist;

			// images
			UChar3Image *source_img;
			Float3Image *cvt_img;
			IntImage *idx_img;

			// superpixel map
			SpixelMap* spixel_map;
			int spixel_size;

			objects::settings gslic_settings;

			virtual void Cvt_Img_Space(UChar3Image* inimg, Float3Image* outimg, COLOR_SPACE color_space) = 0;
			virtual void Init_Cluster_Centers() = 0;
			virtual void Find_Center_Association() = 0;
			virtual void Update_Cluster_Center() = 0;
			virtual void Enforce_Connectivity() = 0;

		public:

			seg_engine(const objects::settings& in_settings );
			virtual ~seg_engine();

			const IntImage* Get_Seg_Mask(bool need_cpu = true) const {
                if  (need_cpu) idx_img->UpdateHostFromDevice();
				return idx_img;
			};

            const SpixelMap* Get_sPixel_Map(bool need_cpu = true) const {
                if (need_cpu) spixel_map->UpdateHostFromDevice();
                return spixel_map;
            }
            
			void Perform_Segmentation(UChar3Image* in_img);
            void Perform_Segmentation(Float3Image* in_img, bool data_on_gpu = true);
			virtual void Draw_Segmentation_Result(UChar3Image* out_img){};
		};
	}
}

