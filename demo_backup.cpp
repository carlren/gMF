#include <time.h>
#include <stdio.h>
#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

#include "gMF_Lib/gMF.h"
#include "NVTimer.h"

using namespace std;
using namespace cv;

void float_to_cv_mat(const float* inimg, Mat& outimg, int w, int h, int dim)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			outimg.at<Vec3b>(y, x)[0] = inimg[idx];
			outimg.at<Vec3b>(y, x)[1] = inimg[idx + 1];
			outimg.at<Vec3b>(y, x)[2] = inimg[idx + 2];
		}
}

void float_to_cv_mat_with_offset(const float* inimg, Mat& outimg, int w, int h, int dim, int offeset)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			outimg.at<Vec3b>(y, x)[0] = inimg[idx + offeset /** 3*/];
			outimg.at<Vec3b>(y, x)[1] = inimg[idx + offeset /** 3*/ + 1];
			outimg.at<Vec3b>(y, x)[2] = inimg[idx + offeset /** 3*/ + 2];
		}
}


void cv_mat_to_float(const Mat& inimg, float* outimg, int w, int h, int dim)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			outimg[idx] = inimg.at<Vec3b>(y, x)[0];
			outimg[idx + 1] = inimg.at<Vec3b>(y, x)[1];
			outimg[idx + 2] = inimg.at<Vec3b>(y, x)[2];
		}
}

void cv_mat_to_float_duplicate(const Mat& inimg, float* outimg, int w, int h, int dim)
{
	int dup_times = dim / 3;

	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			for (int k = 0; k < dup_times;k++)
			{
				outimg[idx + k*3] = inimg.at<Vec3b>(y, x)[0];
				outimg[idx + k * 3 + 1] = inimg.at<Vec3b>(y, x)[1];
				outimg[idx + +k * 3 + 2] = inimg.at<Vec3b>(y, x)[2];
			}
		}
}

void cv_mat_to_distribution(const Mat& inimg, float* outimg, int w, int h, int dim)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;
			const Vec3b& pix = inimg.at<Vec3b>(y, x);
			float sum_val = pix[0] + pix[1] + pix[2] + (10e-3 * 3);
			outimg[idx] = ((float)pix[0] + 25 + 10e-3) / sum_val;
			outimg[idx + 1] = ((float)pix[1] + 20 + 10e-3) / sum_val;
			outimg[idx + 2] = ((float)pix[2] + 10e-3) / sum_val;
		}
}

void distribution_to_cv_mat(const float* inimg, Mat& outimg, int w, int h, int dim)
{
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
		{
			int idx = (x + y * w) * dim;

			if (inimg[idx + 1] > inimg[idx])
			{
				
				if (inimg[idx + 2] > inimg[idx + 1]) outimg.at<Vec3b>(y, x) = Vec3b(255, 0, 0); // b
				else outimg.at<Vec3b>(y, x) = Vec3b(0, 255, 0); // g
			}
			else
			{
				if (inimg[idx + 2] > inimg[idx]) outimg.at<Vec3b>(y, x) = Vec3b(255, 0, 0); // b
				else outimg.at<Vec3b>(y, x) = Vec3b(0, 0, 255); // r
			}
		}
}




int main()
{
	const int dim = 21;
	const int w = 640;
	const int h = 480;

	float* in_array = new float[w * h * dim];
	float* out_array = new float[w * h * dim];

	Mat frame_out; frame_out.create(h, w, CV_8UC3);
	Mat frame_in; frame_in.create(h, w, CV_8UC3);

	Mat frame_small; frame_small.create(240, 320, CV_8UC3);

    Mat frame_old = imread("/home/carl/Work/Code/gmeanfild/data/nature_monte.bmp");
    Mat frame; cv::resize(frame_old,frame,cv::Size(w,h));
	
	cv_mat_to_float_duplicate(frame, in_array, w, h, dim);


	bool keep_playing = true;

	float sigma_bilateral_xy = 48;
	float sigma_bilateral_rgb = 90;
	float sigma_gaussian_xy = 3;
	float w_bilateral = 0.9;
	float w_gaussian = 1 - w_bilateral;

	std::cout << "bilateral_sigma_xy = " << sigma_bilateral_xy<<std::endl;
	std::cout << "bilateral_sigma_rgb = " << sigma_bilateral_rgb << std::endl;
	std::cout << "gaussian_sigma_xy = " << sigma_gaussian_xy << std::endl;
	std::cout << "w_bilateral = " << w_bilateral << std::endl;
	std::cout << "w_gaussian = " << w_gaussian << std::endl;

	gMF::filter_engine *my_filter = new gMF::filter_engine(w, h, dim);
	gMF::BF_info *my_BF_info = new gMF::BF_info(sigma_bilateral_xy, sigma_bilateral_rgb);
	gMF::GF_info *my_GF_info = new gMF::GF_info(sigma_gaussian_xy);
	my_filter->load_reference_image(frame.data, w, h);

	StopWatchInterface *my_timer;
	sdkCreateTimer(&my_timer);
	while (keep_playing)
	{
		
		sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
		my_filter->load_filter_data(in_array, dim, w, h);
        my_filter->filter_bilateral(1, dim, w, h, my_BF_info, false, out_array);
        //my_filter->filter_gaussian(out_array, w_gaussian, dim, w, h, my_GF_info, true);
        sdkStopTimer(&my_timer); printf("\rbilateral filter in:[%.2f]ms ", sdkGetTimerValue(&my_timer));
        std::cout<<std::flush;


		float_to_cv_mat_with_offset(out_array, frame_out, w, h, dim, 0);

		imshow("origin", frame);
		//imshow("in", frame_in);
		imshow("out", frame_out);


		int key = waitKey(10)%256;
		
		switch (key)
		{
		case 27:
			keep_playing = false;
			break;
		case 'C':
			sigma_bilateral_rgb += 5;
			delete my_BF_info;
			my_BF_info = new gMF::BF_info(sigma_bilateral_xy, sigma_bilateral_rgb);

			break;
		case 'c':
			sigma_bilateral_rgb -= 5;
			sigma_bilateral_rgb = max(5.0f, sigma_bilateral_rgb);
			delete my_BF_info;
			my_BF_info = new gMF::BF_info(sigma_bilateral_xy, sigma_bilateral_rgb);

			break;
		case 'X':
			sigma_bilateral_xy += 3;
			delete my_BF_info;
			my_BF_info = new gMF::BF_info(sigma_bilateral_xy, sigma_bilateral_rgb);

			break;
		case'x':
			sigma_bilateral_xy -= 3;
			sigma_bilateral_xy = max(3.0f, sigma_bilateral_xy);
			delete my_BF_info;
			my_BF_info = new gMF::BF_info(sigma_bilateral_xy, sigma_bilateral_rgb);

			break;
		case 'G':
			sigma_gaussian_xy += 1;
			delete my_GF_info;
			my_GF_info = new gMF::GF_info(sigma_gaussian_xy);
			break;
		case'g':
			sigma_gaussian_xy -= 1;
			sigma_gaussian_xy = max(2.0f, sigma_gaussian_xy);
			delete my_GF_info;
			my_GF_info = new gMF::GF_info(sigma_gaussian_xy);
			break;
		case 'b':
			w_bilateral = min(0.9, max(0.1,w_bilateral - 0.1));
			w_gaussian = 1 - w_bilateral;
			break;
		case 'B':
			w_bilateral = min(0.9, max(0.1, w_bilateral + 0.1));
			w_gaussian = 1 - w_bilateral;
			break;
		default:
			break;
		}

		if (key>=0)
		{
			std::cout << std::endl;
			std::cout << "bilateral_sigma_xy = " << sigma_bilateral_xy << std::endl;
			std::cout << "bilateral_sigma_rgb = " << sigma_bilateral_rgb << std::endl;
			std::cout << "gaussian_sigma_xy = " << sigma_gaussian_xy << std::endl;
			std::cout << "w_bilateral = " << w_bilateral << std::endl;
			std::cout << "w_gaussian = " << w_gaussian << std::endl;

			//size_t free_byte, total_byte;
			//cudaMemGetInfo(&free_byte, &total_byte);
			//printf("Free: %lldMB, total: %lldMB\n", free_byte >> 20, total_byte >> 20);
		}

	}

	delete my_filter;
	destroyAllWindows();
    return 0;
}
