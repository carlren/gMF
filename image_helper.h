#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;

int no_colors = 0;
int color_map[255]={0};

int color_to_idx(Vec3b c) {return c[0] + 256*c[1] + 256*256*c[2]+1;}
Vec3b idx_to_color(int cc){ cc--; return Vec3b(cc&0xff, (cc>>8)&0xff, (cc>>16)&0xff);}

void draw_image_from_labeling(Mat& out_img, const int* labling, int w, int h)
{
    for(int y = 0;y<h;y++)
        for (int x=0;x<w;x++)
        {
            int c = color_map[labling[y*w+x]];
            out_img.at<Vec3b>(y,x) = idx_to_color(c);
        }
}

void read_labling_from_image(int* out_labling, const Mat& in_img, int w, int h, int M)
{
    for(int y = 0;y<h;y++)
        for (int x=0;x<w;x++)
        {
            int c = color_to_idx(in_img.at<Vec3b>(y,x));
            int i; for( i=0;i<no_colors && c!=color_map[i]; i++ );
            if(i==no_colors)
            {
                if (i<M) color_map[no_colors++] = c;
                else c=0;
            }
            out_labling[y*w+x] = c?i:-1;
        }
}

void labeling_to_unary(float *out_unary, const int* in_labling, int w, int h, int M)
{
    const float GT_PROB = 0.5;
    const float u_energy = log( 1.0 / M );
	const float n_energy = log( (1.0 - GT_PROB) / (M-1) );
	const float p_energy = log( GT_PROB );
    
    for (int i = 0; i<w*h; i++)
    {
        if(in_labling[i]>=0) 
        {
            for (int k=0;k<M;k++) out_unary[i*M+k] = n_energy;
            out_unary[i*M + in_labling[i]] = p_energy;
        }
        else
        {
            for (int k=0;k<M;k++) out_unary[i*M+k] = u_energy;
        }
    }
}

void Q_dist_to_labeling(int * in_labling, const float *Q_dist, int w, int h, int M)
{
    for (int i=0;i<w*h;i++)
    {
        int max_idx = -1;
        float max_val = -1;
        for (int k=0;k<M;k++)
        {
            if(Q_dist[i*M+k]>max_val)
            {
                max_idx = k;
                max_val = Q_dist[i*M+k];
            }
            
            in_labling[i] = max_idx;
        }
    }
}

void create_pott_compatibility_func(float* in_model, int M)
{
    memset(in_model,0,M*M*sizeof(float));
    for(int i =0; i<M;i++)
    {
        in_model[i*M+i] = -1;
    }
}
